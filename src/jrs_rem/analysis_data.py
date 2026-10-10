"""Shared labeled multimodal inputs for representation analyses."""
from pathlib import Path
import csv
import numpy as np
from .directions import load_features
from .evaluation import labels, is_benign
from .io import sample_key

CATEGORIES = ("benign", "refusal", "jailbreak")


def empty_output(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose an empty analysis output directory")
    return output


def labeled_features(roots, policy="unanimous"):
    seen, shape = set(), None
    for root in roots:
        rows, hs = load_features(root)
        if shape is not None and hs.shape[1:] != shape:
            raise ValueError("Hidden-state shapes differ between input runs")
        shape = hs.shape[1:]
        safe = labels(root, policy) if any(not is_benign(r) for r in rows) else np.ones(len(rows), dtype=int)
        for i, (row, y) in enumerate(zip(rows, safe)):
            key = sample_key(row)
            if key in seen:
                raise ValueError("Duplicate sample identities across analysis inputs")
            seen.add(key)
            if y < 0 or not row.get("img"):
                continue
            category = "benign" if is_benign(row) else ("refusal" if y else "jailbreak")
            yield Path(root), i, row, hs[i], category


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
