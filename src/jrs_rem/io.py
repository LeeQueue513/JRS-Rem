"""Atomic artifacts, stable sample identities, and resumable result storage."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def atomic_array(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.save(f, value, allow_pickle=False)
    os.replace(tmp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def group_key(sample):
    # The same HADES prompt, including its implicit variants, must stay together.
    dataset = {"HADES_hideharm": "HADES", "MM_SafetyBench_hideharm": "MM-SafetyBench"}.get(
        sample["dataset"], sample["dataset"])
    return json.dumps([dataset, str(sample["class"]), str(sample["idx"])], separators=(",", ":"))


def sample_key(sample):
    return json.dumps([sample["dataset"], str(sample["class"]), str(sample["idx"]),
                       sample["data_type"]], separators=(",", ":"))


def validate_samples(rows):
    if not rows:
        raise ValueError("No samples selected")
    keys = []
    for row in rows:
        for field in ("dataset", "class", "idx", "data_type", "txt", "img"):
            if field not in row:
                raise ValueError(f"Missing sample field: {field}")
        if not isinstance(row["txt"], str) or not row["txt"].strip():
            raise ValueError("Sample text must be a nonempty string")
        keys.append(sample_key(row))
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate sample identities")
    return rows


class RunStore:
    """One atomic file per completed sample; resume only identical run inputs.

    A record is committed after its optional hidden-state arrays. Partial/truncated
    JSONL lines and dependence on positional filenames are avoided entirely.
    """
    def __init__(self, root, samples, config):
        self.root = Path(root)
        self.samples = validate_samples(samples)
        self.root.mkdir(parents=True, exist_ok=True)
        self.parts = self.root / "parts"
        self.parts.mkdir(exist_ok=True)
        self.manifest = {"schema_version": 1, "config": config,
                         "samples_sha256": digest(samples), "sample_count": len(samples)}
        path = self.root / "run.json"
        if path.exists() and read_json(path) != self.manifest:
            raise ValueError("Output directory belongs to a different run; choose a new output path")
        if not path.exists():
            atomic_json(path, self.manifest)

    def stem(self, sample):
        return digest(sample_key(sample))

    def done(self, sample):
        stem = self.stem(sample)
        if not (self.parts / (stem + ".json")).exists():
            return False
        if self.manifest["config"].get("save_hidden"):
            return all((self.parts / (stem + suffix)).exists() for suffix in (".hs.npy", ".text.npy"))
        return True

    def save(self, sample, answer, hs=None, text_hs=None):
        stem = self.stem(sample)
        if self.manifest["config"].get("save_hidden"):
            if hs is None or text_hs is None:
                raise ValueError("Both multimodal and text hidden states are required")
            atomic_array(self.parts / (stem + ".hs.npy"), np.asarray(hs, dtype=np.float32))
            atomic_array(self.parts / (stem + ".text.npy"), np.asarray(text_hs, dtype=np.float32))
        atomic_json(self.parts / (stem + ".json"), {**sample, "output_txt": answer})

    def finalize(self):
        if not all(self.done(s) for s in self.samples):
            raise ValueError("Cannot finalize an incomplete run")
        records = [read_json(self.parts / (self.stem(s) + ".json")) for s in self.samples]
        if self.manifest["config"].get("save_hidden"):
            for suffix, filename in ((".hs.npy", "hs.npy"), (".text.npy", "text_hs.npy")):
                first = np.load(self.parts / (self.stem(self.samples[0]) + suffix), allow_pickle=False)
                tmp = self.root / (filename + ".tmp")
                arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32,
                                               shape=(len(self.samples), *first.shape))
                for i, s in enumerate(self.samples):
                    value = np.load(self.parts / (self.stem(s) + suffix), allow_pickle=False)
                    if value.shape != first.shape:
                        raise ValueError("Hidden-state shapes differ across samples")
                    arr[i] = value
                arr.flush()
                del arr
                os.replace(tmp, self.root / filename)
        atomic_json(self.root / "results.json", records)
        # Retain parts so a resumed run can verify completion without rerunning GPU work.
        return records
