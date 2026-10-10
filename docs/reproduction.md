# Experiments

Start with the [README workflow](../README.md#quick-start). Run commands from the repository root; see [data preparation](data.md) for inputs.

## Settings

- Generation uses greedy decoding with at most 128 new tokens by default. The example configs use seed 0 and float16.
- At each layer, JRS-Rem removes the signed projection of `h_image_text - h_text` onto the jailbreak direction when cosine alignment is at least `threshold` (default 0.2). Removal happens once at the last prompt token during prefill. Text-only inputs use baseline generation.
- Keep model, inputs, `prefix`, and `max_new_tokens` identical across baseline/defense comparisons, and supply the same `direction` file to exclude calibration prompts from both.
- For another dataset, run `prepare` and copy a config with new `samples` and `output` paths. Reuse the HADES direction for cross-dataset evaluation.
- `strength` controls projection removal (default 1.0; 0 provides a control). Use separate output directories for different settings. Rerun the same command with unchanged inputs to resume.

## Safety evaluation

Three judges vote on each answer: refusal matching (`machine`), Qwen3Guard-Gen-8B, and Llama-Guard-4-12B. The model judges receive the answer only. Qwen's `Controversial` label counts as unsafe.

ASR uses majority voting and reports unsafe/total counts and fractions, overall and by data type. Direction estimation, PCA/distance analysis, and probes use unanimous labels by default, excluding disagreements. Standalone shift and warning analysis default to majority voting. Keep label policies consistent across comparisons.

## Utility evaluation

Run matched baseline/defense inference on a benign dataset, then score each output:

```bash
jrs-rem utility --results outputs/scienceqa --dataset ScienceQA --data-root /path/to/datasets
```

The command takes one benchmark per output directory. MME reports per-category accuracy plus paired-image accuracy, each multiplied by 100. ScienceQA, RealWorldQA, and single-image MMMU report accuracy using the local answer parsers in [utility.py](../src/jrs_rem/utility.py). MM-Vet inputs are supported for generation; scoring requires an external MM-Vet grader.

## Representation analysis

Collect harmful and benign baseline outputs with `save_hidden: true`, using the same checkpoint and input settings. Judge harmful answers with all three judges; benign manifests use `stxt` as described in the [sample schema](data.md#sample-manifest). Saved representations are always from before intervention.

```bash
jrs-rem analyze --results outputs/baseline outputs/benign --layer 19 --output outputs/analysis --plot
jrs-rem probe-three-class --results outputs/baseline outputs/benign --epochs 10 --seed 42 --output outputs/three_class
jrs-rem probe --results outputs/baseline outputs/benign --layer 19 --epochs 10 --seed 0 --output outputs/probe
jrs-rem warnings --results outputs/baseline --output outputs/warnings
jrs-rem shift --results outputs/baseline outputs/benign --direction outputs/direction.npz --output outputs/shift
```

- **PCA and distances (`analyze`):** exports PCA coordinates, cosine distances to each layer's mean jailbreak center, and mean/std summaries; `--plot` adds figures. Keep the input pool and layer fixed for PCA comparisons.
- **Three-class probe:** classifies benign/refusal/jailbreak states per layer. It uses a shared 80/20 prompt-group split and caps each class at 2,000 training and 500 test samples (`--train-cap`, `--test-cap`). Use `--layers` to select layers; by default all indices, including embedding 0, are included. Outputs include per-layer F1, predictions, and weights.
- **Binary probe (`probe`):** trains on benign/refusal samples and reports held-out accuracy and mean harmfulness probability on unseen jailbreak prompts. Both probes split by prompt group to keep variants together, use Adam (1e-4, batch size 64), and evaluate the final epoch without test-based model selection.
- **Safety warnings:** counts responses matching the paper's warning phrases among **successful multimodal jailbreaks only**. Hidden states are not needed.
- **Shift analysis:** exports signed projections and cosine alignment of paired image/text representations along the direction, with per-layer group summaries. To restrict analysis to text-safe inputs, separately generate and judge matching text-only answers, then add `--text-results outputs/text_only`. Saved `text_hs.npy` alone does not provide safety labels.

Use `jrs-rem <command> --help` for additional options.
