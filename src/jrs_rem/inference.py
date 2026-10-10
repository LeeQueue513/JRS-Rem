"""One LLaVA runner for baseline and defense."""
from pathlib import Path
from .io import read_json, validate_samples, group_key, RunStore, file_digest
from .directions import load_direction

DEFAULTS = {
    "model": "llava-hf/llava-1.5-7b-hf", "revision": None,
    "device": "cuda", "dtype": "float16", "max_new_tokens": 128,
    "prefix": "", "save_hidden": False, "mode": "baseline", "direction": None,
    "threshold": 0.2, "strength": 1.0, "projection": "signed",
    "seed": 0, "limit": 0,
}


def run(config_path):
    config_path = Path(config_path).resolve()
    raw = read_json(config_path)
    unknown = set(raw) - set(DEFAULTS) - {"samples", "output"}
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    cfg = {**DEFAULTS, **raw}
    if cfg["projection"] != "signed":
        raise ValueError("Only projection=signed is supported; use a new output directory")
    for name in ("samples", "output"):
        if not cfg.get(name):
            raise ValueError(f"Config requires {name}")
    for name in ("samples", "output", "direction"):
        if cfg.get(name):
            cfg[name] = str((config_path.parent / cfg[name]).resolve())
    if str(cfg["model"]).startswith((".", "/")):
        cfg["model"] = str((config_path.parent / cfg["model"]).resolve())
    if cfg["mode"] not in ("baseline", "defense"):
        raise ValueError("mode must be baseline or defense")
    if cfg["max_new_tokens"] < 1 or cfg["limit"] < 0 or cfg["strength"] < 0:
        raise ValueError("Invalid generation length, limit, or strength")
    if not -1 <= cfg["threshold"] <= 1:
        raise ValueError("threshold must be between -1 and 1")
    rows = validate_samples(read_json(cfg["samples"]))
    for r in rows:
        if r.get("img"):
            r["img"] = str((Path(cfg["samples"]).parent / r["img"]).resolve())
            if not Path(r["img"]).is_file():
                raise FileNotFoundError(r["img"])
    ds = meta = None
    # Baseline can also receive a direction to evaluate exactly the same held-out pool.
    if cfg["direction"]:
        ds, meta = load_direction(cfg["direction"])
        excluded = set(meta["calibration_groups"])
        rows = [r for r in rows if group_key(r) not in excluded]
        cfg["direction_sha256"] = file_digest(cfg["direction"])
    elif cfg["mode"] != "baseline":
        raise ValueError("A direction with recorded calibration IDs is required")
    if cfg["limit"]:
        rows = rows[:cfg["limit"]]
    validate_samples(rows)
    # Fingerprint image contents so same-path image edits cannot silently resume old outputs.
    cfg["images_sha256"] = {p: file_digest(p) for p in sorted({r["img"] for r in rows if r.get("img")})}
    import torch
    import transformers
    from .llava import load_model, build_input, hidden_states, generate
    from .defense import Intervention, correction
    from . import __version__
    cfg["runtime"] = {"torch": torch.__version__, "transformers": transformers.__version__, "jrs_rem": __version__}
    store = RunStore(cfg["output"], rows, cfg)
    if all(store.done(row) for row in rows):
        store.finalize()
        return
    torch.manual_seed(cfg["seed"])
    model, processor = load_model(cfg["model"], cfg["device"], cfg["dtype"], cfg["revision"])
    if ds is not None:
        ds = torch.from_numpy(ds).to(cfg["device"])
    for i, row in enumerate(rows):
        if store.done(row):
            continue
        inputs = build_input(processor, row, cfg["device"], prefix=cfg["prefix"])
        multi = text = None
        if cfg["save_hidden"] or cfg["mode"] == "defense":
            multi = hidden_states(model, inputs)
            text = hidden_states(model, build_input(processor, row, cfg["device"], text_only=True, prefix=cfg["prefix"])) if row.get("img") else multi
        if cfg["mode"] == "defense" and row.get("img"):
            if multi.shape != ds.shape:
                raise ValueError("Direction and model hidden states have different shapes")
            vector, active = correction(text, multi, ds, cfg["threshold"], cfg["projection"])
            context = Intervention(model, -cfg["strength"] * vector, active)
        else:
            from contextlib import nullcontext
            context = nullcontext()
        with context:
            answer = generate(model, processor, inputs, cfg["max_new_tokens"])
        store.save(row, answer, multi.cpu().numpy() if multi is not None else None,
                   text.cpu().numpy() if text is not None else None)
        print(f"Completed {i+1}/{len(rows)}", flush=True)
    store.finalize()
