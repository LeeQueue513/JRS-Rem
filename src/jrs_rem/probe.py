"""Benign/refusal linear probe with prompt-group-held-out evaluation."""
from pathlib import Path
import numpy as np
from .directions import load_features
from .evaluation import labels, is_benign
from .io import group_key, atomic_json


def group_split(rows, seed, ratio=.8):
    rng = np.random.default_rng(seed)
    groups = sorted({group_key(r) for r in rows})
    if len(groups) < 2:
        raise ValueError("Need at least two independent prompt groups")
    rng.shuffle(groups)
    cut = min(len(groups)-1, max(1, int(len(groups)*ratio)))
    train = set(groups[:cut])
    return np.asarray([group_key(r) in train for r in rows])


def train(roots, output, layer=19, epochs=10, seed=0, device="cpu"):
    import torch
    from torch import nn
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty probe output directory")
    torch.manual_seed(seed)
    rows, x, categories = [], [], []
    for root in roots:
        records, hs = load_features(root)
        safe = labels(root, "unanimous") if any(not is_benign(r) for r in records) else np.ones(len(records), dtype=int)
        if not 0 <= layer < hs.shape[1]:
            raise ValueError("Layer outside hidden-state range")
        for i, (r, y) in enumerate(zip(records, safe)):
            if y < 0:
                continue
            rows.append(r)
            x.append(hs[i, layer])
            categories.append(0 if is_benign(r) else (1 if y else 2))
    train_mask = group_split(rows, seed)
    categories = np.asarray(categories)
    fit = train_mask & (categories < 2)
    test = ~train_mask & (categories < 2)
    jailbreak = ~train_mask & (categories == 2)
    if set(categories[fit]) != {0, 1} or set(categories[test]) != {0, 1}:
        raise ValueError("Both benign and refusal classes must occur in train and held-out sets")
    features = torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
    targets = torch.tensor(categories.clip(max=1), dtype=torch.long, device=device)
    model = nn.Linear(features.shape[-1], 2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    indices = np.flatnonzero(fit)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(indices)
        for start in range(0, len(indices), 64):
            ids = torch.tensor(indices[start:start+64], device=device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(features[ids]), targets[ids])
            loss.backward()
            optimizer.step()
    with torch.inference_mode():
        probabilities = model(features).softmax(-1)[:, 1].cpu().numpy()
    accuracy = float(((probabilities[test] >= .5) == categories[test]).mean())
    output.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / "probe.pt")
    result = {"layer": layer, "epochs": epochs, "seed": seed, "train_count": int(fit.sum()),
              "test_count": int(test.sum()), "test_accuracy": accuracy,
              "unseen_jailbreak_count": int(jailbreak.sum()),
              "unseen_jailbreak_mean_harmful_probability": float(probabilities[jailbreak].mean()) if jailbreak.any() else None,
              "train_groups": sorted({group_key(r) for r, t in zip(rows, train_mask) if t}),
              "test_groups": sorted({group_key(r) for r, t in zip(rows, train_mask) if not t}),
              "protocol": "fixed-final-epoch; no checkpoint selection on held-out test"}
    atomic_json(output / "probe.json", result)
    return result


def class_metrics(truth, predicted, count=3):
    """Per-class held-out metrics; an undefined precision/recall is zero."""
    metrics = []
    for c in range(count):
        tp = int(((truth == c) & (predicted == c)).sum())
        support = int((truth == c).sum())
        positives = int((predicted == c).sum())
        precision = tp / positives if positives else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + positives) if support + positives else 0.0
        metrics.append({'precision': precision, 'recall': recall, 'f1': f1, 'support': support})
    return metrics


def train_three_class(roots, output, layers=None, epochs=10, seed=42, device='cpu',
                      train_cap=2000, test_cap=500, policy='unanimous'):
    """Fixed-final-epoch linear probes; one shared prompt split for all layers."""
    import torch
    from torch import nn
    from .analysis_data import labeled_features, empty_output, write_csv, CATEGORIES
    from .io import sample_key
    output = empty_output(output)
    if epochs < 1 or train_cap < 1 or test_cap < 1:
        raise ValueError('Epochs and per-class sample caps must be positive')
    data = list(labeled_features(roots, policy))
    if not data:
        raise ValueError('No labeled multimodal samples')
    rows = [d[2] for d in data]
    targets_np = np.array([CATEGORIES.index(d[4]) for d in data])
    train_mask = group_split(rows, seed)
    rng = np.random.default_rng(seed)

    def balanced_indices(mask, cap):
        chunks = []
        for c in range(3):
            candidates = np.flatnonzero(mask & (targets_np == c))
            if not len(candidates):
                raise ValueError('All three classes must occur in both prompt-group splits')
            chunks.append(rng.choice(candidates, min(cap, len(candidates)), replace=False))
        return np.concatenate(chunks)

    fit = balanced_indices(train_mask, train_cap)
    test = balanced_indices(~train_mask, test_cap)
    n_layers = data[0][3].shape[0]
    layers = list(range(n_layers)) if layers is None else list(layers)
    if not layers or len(set(layers)) != len(layers) or any(l < 0 or l >= n_layers for l in layers):
        raise ValueError('Select unique layers in the hidden-state range')
    torch.manual_seed(seed)
    targets = torch.tensor(targets_np[fit], dtype=torch.long, device=device)
    reports, predictions = [], []
    # Train and evaluate one layer at a time to avoid allocating all features on GPU.
    output.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        xfit = torch.tensor(np.asarray([data[i][3][layer] for i in fit]), dtype=torch.float32, device=device)
        xtest = torch.tensor(np.asarray([data[i][3][layer] for i in test]), dtype=torch.float32, device=device)
        model = nn.Linear(xfit.shape[-1], 3).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        for _ in range(epochs):
            order = rng.permutation(len(fit))
            for start in range(0, len(fit), 64):
                index = torch.tensor(order[start:start+64], device=device)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(xfit[index]), targets[index])
                loss.backward()
                optimizer.step()
        with torch.inference_mode():
            predicted = model(xtest).argmax(-1).cpu().numpy()
        metrics = class_metrics(targets_np[test], predicted)
        reports.append({'layer': layer, 'accuracy': float((predicted == targets_np[test]).mean()),
                        'classes': dict(zip(CATEGORIES, metrics))})
        predictions.extend([sample_key(rows[i]), layer, CATEGORIES[targets_np[i]], CATEGORIES[p]]
                           for i, p in zip(test, predicted))
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, output / f'probe_layer_{layer}.pt')
    write_csv(output / 'f1_by_layer.csv', ['layer', 'label', 'precision', 'recall', 'f1', 'support'],
              ([r['layer'], c, *r['classes'][c].values()] for r in reports for c in CATEGORIES))
    write_csv(output / 'predictions.csv', ['sample_key', 'layer', 'truth', 'prediction'], predictions)
    result = {'roots': list(map(str, roots)), 'policy': policy, 'seed': seed, 'epochs': epochs,
              'layers': layers, 'layer_zero': 'embedding', 'train_cap_per_class': train_cap,
              'test_cap_per_class': test_cap, 'train_samples': [sample_key(rows[i]) for i in fit],
              'test_samples': [sample_key(rows[i]) for i in test],
              'train_groups': sorted({group_key(rows[i]) for i in fit}),
              'test_groups': sorted({group_key(rows[i]) for i in test}),
              'protocol': '80/20 prompt-group split; per-class caps; fixed final epoch; no test selection',
              'optimizer': 'Adam', 'learning_rate': 1e-4, 'batch_size': 64, 'reports': reports}
    atomic_json(output / 'probe.json', result)
    return result
