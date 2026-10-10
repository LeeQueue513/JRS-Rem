"""Paired image-induced shifts; no plotting offsets or modality-direction removal."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from .analysis_data import labeled_features, empty_output, write_csv
from .directions import load_direction, projection
from .evaluation import labels, is_benign
from .io import read_json, sample_key, atomic_json


def text_key(row):
    # Exact text distinguishes implicit rewrites sharing the same prompt group.
    return (row['dataset'], str(row['class']), str(row['idx']), row['txt'])


def analyze_shifts(roots, output, direction, policy="unanimous", text_results=None):
    output = empty_output(output)
    ds, _ = load_direction(direction)
    text_safe = {}
    if text_results:
        for root in text_results:
            rows = read_json(Path(root) / 'results.json')
            ys = labels(root, policy) if any(not is_benign(r) for r in rows) else np.ones(len(rows), dtype=int)
            for r, y in zip(rows, ys):
                if r.get('img'):
                    raise ValueError("--text-results requires text-only runs")
                key = text_key(r)
                if key in text_safe:
                    raise ValueError("Duplicate text-only pairing keys")
                text_safe[key] = int(y)
    groups, records, cache = defaultdict(list), [], {}
    skipped = 0
    for root, i, row, hs, category in labeled_features(roots, policy):
        if text_results:
            key = text_key(row)
            if key not in text_safe:
                raise ValueError("Missing text-only judgment for a multimodal sample")
            if text_safe[key] != 1:
                skipped += 1
                continue
        if root not in cache:
            text = np.load(root / 'text_hs.npy', mmap_mode='r', allow_pickle=False)
            full = np.load(root / 'hs.npy', mmap_mode='r', allow_pickle=False)
            if text.shape != full.shape or not np.isfinite(text).all():
                raise ValueError("Invalid paired text hidden states")
            cache[root] = text
        if ds.shape != hs.shape:
            raise ValueError("Direction/feature shape mismatch")
        shift = np.asarray(hs, dtype=np.float32) - cache[root][i]
        cosine, _ = projection(shift, ds)
        scalar = (shift * ds).sum(-1)
        # A zero shift has zero projection; cosine is defined as zero by convention.
        values = np.stack([cosine, scalar], axis=-1)
        groups[(row['dataset'], row['data_type'], category)].append(values)
        for l, (c, s) in enumerate(values):
            records.append([sample_key(row), row['dataset'], row['data_type'], category, l, float(c), float(s)])
    if not records:
        raise ValueError("No labeled paired multimodal samples remain")
    summary = []
    for key, values in sorted(groups.items()):
        a = np.asarray(values)
        for l, (mean, std) in enumerate(zip(a.mean(0), a.std(0))):
            summary.append([*key, l, len(a), float(mean[0]), float(std[0]), float(mean[1]), float(std[1])])
    write_csv(output / 'shifts.csv', ['sample_key', 'dataset', 'data_type', 'label', 'layer', 'cosine', 'signed_projection'], records)
    write_csv(output / 'shift_summary.csv', ['dataset', 'data_type', 'label', 'layer', 'count', 'cosine_mean', 'cosine_std', 'projection_mean', 'projection_std'], summary)
    result = {'roots': list(map(str, roots)), 'direction': str(direction), 'policy': policy,
              'text_results': list(map(str, text_results or [])), 'require_text_safe': bool(text_results),
              'excluded_text_unsafe_or_conflict': skipped, 'std_ddof': 0, 'zero_shift_cosine': 0,
              'layer_zero': 'embedding', 'paired_samples': sum(map(len, groups.values()))}
    atomic_json(output / 'shift.json', result)
    return result
