"""Direction estimation from local baseline representations and recorded judges."""
import json
import random
from pathlib import Path
import numpy as np
from .io import read_json, atomic_json, group_key, sample_key, file_digest
from .evaluation import labels, is_benign


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-8)


def center(x):
    # Paper definition, confirmed by the authors: arithmetic mean per layer.
    return np.asarray(x, dtype=np.float32).mean(axis=0)


def projection(force, direction, mode="signed"):
    if mode != "signed":
        raise ValueError("Only signed projection is supported")
    force = np.asarray(force, dtype=np.float32)
    direction = unit(direction)
    scalar = np.sum(force * direction, axis=-1)
    cosine = scalar / np.maximum(np.linalg.norm(force, axis=-1), 1e-8)
    vector = scalar[..., None] * direction
    return cosine, vector


def load_features(root):
    root = Path(root)
    rows = read_json(root / "results.json")
    hs = np.load(root / "hs.npy", mmap_mode="r", allow_pickle=False)
    if hs.ndim != 3 or len(hs) != len(rows):
        raise ValueError("Expected hs.npy shape [samples, embedding+decoder layers, width]")
    if not np.isfinite(hs).all():
        raise ValueError("Hidden states contain NaN/Inf")
    return rows, hs


def select_calibration(rows, safe, count, seed):
    """Choose one variant per prompt and disjoint refusal/jailbreak prompt groups."""
    if count < 1:
        raise ValueError("Calibration count must be positive")
    rng = random.Random(seed)
    used, selected = set(), []
    for label in (1, 0):
        pool = [i for i, (r, y) in enumerate(zip(rows, safe))
                if y == label and not is_benign(r) and r.get("img") and group_key(r) not in used]
        rng.shuffle(pool)
        chosen = []
        for i in pool:
            key = group_key(rows[i])
            if key not in used:
                chosen.append(i)
                used.add(key)
            if len(chosen) == count:
                break
        if len(chosen) != count:
            raise ValueError(f"Need {count} unique prompt groups for label {label}; found {len(chosen)}")
        selected.append(chosen)
    return selected


def estimate(root, output, count=50, seed=0, policy="unanimous"):
    root, output = Path(root), Path(output)
    if output.suffix != ".npz":
        raise ValueError("Direction output must end in .npz")
    if output.exists():
        raise FileExistsError(output)
    rows, hs = load_features(root)
    safe = labels(root, policy)
    refusal, jailbreak = select_calibration(rows, safe, count, seed)
    ds = unit(center(hs[jailbreak]) - center(hs[refusal]))
    calibration = sorted(set(group_key(rows[i]) for i in refusal + jailbreak))
    meta = {"schema_version": 1, "model_family": "llava", "center": "mean",
            "modality_removal": False, "judge_policy": policy, "count_per_label": count,
            "seed": seed, "results_sha256": file_digest(root / "results.json"),
            "calibration_groups": calibration,
            "refusal_samples": [sample_key(rows[i]) for i in refusal],
            "jailbreak_samples": [sample_key(rows[i]) for i in jailbreak],
            "evaluation_count": sum(group_key(r) not in calibration for r in rows)}
    arrays = {"ds": ds}
    arrays["metadata"] = np.asarray(json.dumps(meta))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **arrays)
    output.with_suffix(".tmp").replace(output)
    atomic_json(output.with_suffix(".json"), meta)
    return meta


def load_direction(path):
    with np.load(path, allow_pickle=False) as a:
        ds = np.array(a["ds"], dtype=np.float32)
        meta = json.loads(str(a["metadata"]))
    if ds.ndim != 2 or not np.isfinite(ds).all():
        raise ValueError("Invalid direction array")
    if not isinstance(meta.get("calibration_groups"), list):
        raise ValueError("Direction must include calibration prompt identities")
    return unit(ds), meta
