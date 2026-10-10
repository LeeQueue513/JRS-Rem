"""PCA and layer-wise cosine distance to the arithmetic-mean jailbreak center."""
import numpy as np
from .io import atomic_json, sample_key
from .directions import center, unit
from .analysis_data import labeled_features, empty_output, write_csv, CATEGORIES


def cosine_distances(features, categories):
    mask = np.asarray(categories) == "jailbreak"
    if not mask.any():
        raise ValueError("No jailbreak samples remain under the chosen judge policy")
    centroid = center(features[mask])
    if (np.linalg.norm(features, axis=-1) < 1e-8).any() or (np.linalg.norm(centroid, axis=-1) < 1e-8).any():
        raise ValueError("Cosine distance is undefined for a zero representation or centroid")
    return 1 - np.clip((unit(features) * unit(centroid)).sum(-1), -1, 1)


def analyze(roots, output, layer=19, policy="unanimous", direction=None, plot=False):
    output = empty_output(output)
    data = list(labeled_features(roots, policy))
    if len(data) < 3:
        raise ValueError("At least three labeled multimodal samples are required")
    rows = [d[2] for d in data]
    features = np.asarray([d[3] for d in data], dtype=np.float32)
    categories = np.asarray([d[4] for d in data])
    if not 0 <= layer < features.shape[1]:
        raise ValueError("Layer outside hidden-state range")
    distances = cosine_distances(features, categories)
    x = features[:, layer]
    u, singular, _ = np.linalg.svd(x - x.mean(0), full_matrices=False)
    xy = u[:, :2] * singular[:2]
    write_csv(output / "representations.csv",
              ["sample_key", "dataset", "class", "idx", "data_type", "label", "pc1", "pc2", "cosine_distance_to_jailbreak"],
              ([sample_key(r), r["dataset"], r["class"], r["idx"], r["data_type"], c, *v, float(d)]
               for r, c, v, d in zip(rows, categories, xy, distances[:, layer])))
    write_csv(output / "distances.csv", ["sample_key", "label", "layer", "cosine_distance"],
              ([sample_key(r), c, l, float(d)] for r, c, values in zip(rows, categories, distances)
               for l, d in enumerate(values)))
    summary = []
    for c in CATEGORIES:
        values = distances[categories == c]
        if len(values):
            for l, (mean, std) in enumerate(zip(values.mean(0), values.std(0))):
                summary.append([c, l, len(values), float(mean), float(std)])
    write_csv(output / "distance_summary.csv", ["label", "layer", "count", "mean", "std"], summary)
    atomic_json(output / "analysis.json", {"roots": [str(r) for r in roots], "pca_layer": layer,
                "policy": policy, "distance": "cosine", "center": "arithmetic_mean",
                "std_ddof": 0, "multimodal_only": True, "layer_zero": "embedding",
                "counts": {c: int((categories == c).sum()) for c in CATEGORIES}})
    if direction:
        from .shifts import analyze_shifts
        analyze_shifts(roots, output / "shift", direction, policy)
    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in CATEGORIES:
            mask = categories == c
            if mask.any():
                ax.scatter(*xy[mask].T, label=c, s=10, alpha=.6)
        ax.set(xlabel="PC1", ylabel="PC2", title=f"Representations, layer {layer}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "pca.svg")
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        for c in CATEGORIES:
            values = distances[categories == c]
            if len(values):
                mean, std = values.mean(0), values.std(0)
                line, = ax.plot(mean, label=c)
                ax.fill_between(range(len(mean)), mean-std, mean+std, color=line.get_color(), alpha=.15)
        ax.set(xlabel="Layer (0 = embedding)", ylabel="Cosine distance to jailbreak center")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "distances.svg")
        plt.close(fig)
