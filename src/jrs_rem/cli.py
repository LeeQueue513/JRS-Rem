"""All public entry points; --help does not require GPU dependencies."""
import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="jrs-rem", description="LLaVA JRS-Rem inference, evaluation and analysis")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Convert an existing dataset directory to a sample manifest")
    p.add_argument("--dataset", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--hades-step", type=int, default=4)
    p = sub.add_parser("infer", help="Run baseline or defense from JSON config")
    p.add_argument("--config", required=True)
    p = sub.add_parser("judge", help="Judge answers using one of the three existing judges")
    p.add_argument("--results", required=True)
    p.add_argument("--judge", required=True, choices=["machine", "Qwen3Guard-Gen-8B", "Llama-Guard-4-12B"])
    p.add_argument("--model", help="Judge checkpoint directory or Hugging Face ID")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    p = sub.add_parser("asr", help="Report three-vote ASR with sample counts")
    p.add_argument("--results", required=True)
    p = sub.add_parser("direction", help="Estimate a direction and record calibration prompt IDs")
    p.add_argument("--results", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--count", type=int, default=50, help="Number of unique prompts per label")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy", choices=["majority", "unanimous"], default="unanimous")
    p = sub.add_parser("analyze", help="Export PCA, distances, and paired shift distributions")
    p.add_argument("--results", required=True, nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--policy", choices=["majority", "unanimous"], default="unanimous")
    p.add_argument("--direction")
    p.add_argument("--plot", action="store_true")
    p = sub.add_parser("probe", help="Train a benign/refusal linear probe with a prompt-group split")
    p.add_argument("--results", required=True, nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--layer", type=int, default=19)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p = sub.add_parser("probe-three-class", help="Per-layer benign/refusal/jailbreak linear probes")
    p.add_argument("--results", required=True, nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--layers", type=int, nargs="+", help="Default: all layers, including embedding 0")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--train-cap", type=int, default=2000)
    p.add_argument("--test-cap", type=int, default=500)
    p.add_argument("--policy", choices=["majority", "unanimous"], default="unanimous")
    p = sub.add_parser("warnings", help="Count safety warnings among successful multimodal jailbreaks")
    p.add_argument("--results", required=True, nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--policy", choices=["majority", "unanimous"], default="majority")
    p = sub.add_parser("shift", help="Export paired shift values and per-layer group summaries")
    p.add_argument("--results", required=True, nargs="+")
    p.add_argument("--output", required=True)
    p.add_argument("--direction", required=True)
    p.add_argument("--policy", choices=["majority", "unanimous"], default="majority")
    p.add_argument("--text-results", nargs="+", help="Judged text-only runs; restrict pairs to text-safe inputs")
    p = sub.add_parser("utility", help="Score one benign benchmark using prepared ground truth")
    p.add_argument("--results", required=True)
    p.add_argument("--dataset", required=True, choices=["MME", "ScienceQA", "RealWorldQA", "MMMU"])
    p.add_argument("--data-root", required=True)
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, ImportError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    if result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2))


def dispatch(a):
    if a.command == "prepare":
        from .data import prepare
        rows = prepare(a.dataset, a.data_root, a.output, a.hades_step)
        return {"samples": len(rows), "manifest": a.output}
    if a.command == "infer":
        from .inference import run
        return run(a.config)
    if a.command == "judge":
        from .evaluation import judge
        result = judge(a.results, a.judge, a.model, a.device, a.dtype)
        return {"judge": a.judge, "count": len(result)}
    if a.command == "asr":
        from .evaluation import summarize
        return summarize(a.results)
    if a.command == "direction":
        from .directions import estimate
        return estimate(a.results, a.output, a.count, a.seed, a.policy)
    if a.command == "analyze":
        from .analysis import analyze
        analyze(a.results, a.output, a.layer, a.policy, a.direction, a.plot)
        return {"output": a.output}
    if a.command == "probe":
        from .probe import train
        if a.epochs < 1:
            raise ValueError("epochs must be positive")
        return train(a.results, a.output, a.layer, a.epochs, a.seed, a.device)
    if a.command == "probe-three-class":
        from .probe import train_three_class
        return train_three_class(a.results, a.output, a.layers, a.epochs, a.seed, a.device,
                                 a.train_cap, a.test_cap, a.policy)
    if a.command == "warnings":
        from .warnings import analyze_warnings
        return analyze_warnings(a.results, a.output, a.policy)
    if a.command == "shift":
        from .shifts import analyze_shifts
        return analyze_shifts(a.results, a.output, a.direction, a.policy, a.text_results)
    if a.command == "utility":
        from . import utility
        from .io import read_json, atomic_json, sample_key
        rows = read_json(Path(a.results) / "results.json")
        if not rows or any(r["dataset"] != a.dataset for r in rows):
            raise ValueError("Provide a nonempty single-benchmark result directory")
        if len({sample_key(r) for r in rows}) != len(rows):
            raise ValueError("Duplicate sample identities")
        fn = {"MME": utility.eval_mme, "ScienceQA": utility.eval_scienceqa,
              "RealWorldQA": utility.eval_realworldqa, "MMMU": utility.eval_mmmu}[a.dataset]
        value = fn(rows, a.data_root)
        keys = {"MME": ["perception", "cognition", "per_category", "unparsed"],
                "ScienceQA": ["accuracy", "per_split"],
                "RealWorldQA": ["accuracy", "per_split", "unparsed"],
                "MMMU": ["accuracy", "per_subject", "per_type", "unparsed"]}[a.dataset]
        result = {"dataset": a.dataset, "count": len(rows), **dict(zip(keys, value))}
        atomic_json(Path(a.results) / "utility.json", result)
        return result
