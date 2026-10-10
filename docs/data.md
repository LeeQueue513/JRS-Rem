# Data and outputs

## Dataset preparation

`jrs-rem prepare` reads existing datasets; it does not download them. Expected files under `--data-root`:

| Dataset | Files |
|---|---|
| HADES | `HADES/instructions/{Animal,Financial,Privacy,Self-Harm,Violence}.json`; `HADES/optimized_SD_images/{class}/{id}/{opt_N,concat_opt_N,keyword}.jpg` |
| HADES_hideharm | HADES plus `HADES/MML/{class}/data.json` with matching instruction and `hades_prompt` |
| MM-SafetyBench | `MM-SafetyBench/processed_questions/*.json`; `MM-SafetyBench/imgs/{class}/{SD,SD_TYPO,TYPO}/{index}.jpg` |
| MM_SafetyBench_hideharm | Same source, `Rephrased Question` with SD_TYPO image |
| RedTeam_2K | `RedTeam_2K/RedTeam_2K_filtered.csv` (`id`, `question`, `policy`); `RedTeam_2K/imgs/{type}_related_{id}.png` |
| MMvet | `MMvet/mm-vet.json`; `MMvet/images/` |
| MME | `MME/MME.json` (`question_id`, `category`, `question`, `img`, `answer`) |
| ScienceQA | `ScienceQA/test_data.json` (`idx`, `cls`, `question`, `img`, `choices`, `answer`) |
| RealWorldQA | `RealWorldQA/realworldqa.json` (`idx`, `question`, `answer`); `RealWorldQA/images/{idx:04d}.png` |
| MMMU | `MMMU/mmmu_val.json` (`idx`, `subject`, `question`, `qtype`, `options`, `answer`); `MMMU/images/{idx}.png` |

HADES defaults to image optimization step 4 (`--hades-step`). MM-SafetyBench uses `Changed Question` for explicit prompts and `Rephrased Question` for implicit prompts; preserve JSON question order for image indexing.

MMMU uses the prepared **single-image validation subset**. ScienceQA answers are zero-based choice indices. MME requires both questions per image.

## Sample manifest

Custom inputs, including benign LLaVA-Instruct data, can use a JSON list directly:

```json
[
  {
    "dataset": "example",
    "class": "example_category",
    "idx": "example_001",
    "data_type": "stxt+rimg",
    "txt": "Describe the colors in this picture.",
    "img": "images/example_001.png"
  }
]
```

Image paths are absolute or relative to the manifest; `img: null` means text-only. Use `stxt` only for benign data: these rows are automatically labeled safe and excluded from ASR. Explicit and implicit harmful inputs use `utxt+uimg[...]` and `m_utxt+uimg[...]`, respectively.

Samples are identified by `(dataset, class, idx, data_type)` and grouped by `(dataset, class, idx)` for calibration and probe splits. HADES and MM-SafetyBench implicit variants share groups with their explicit counterparts.

## Outputs

- `results.json`: input records and generated `output_txt`.
- `run.json` and `parts/`: run settings and per-sample files for resuming inference; retain them when resuming.
- `hs.npy`, `text_hs.npy`: optional paired **pre-intervention** representations at the last prompt token, shaped `[samples, 33, 4096]` for LLaVA-1.5-7B. Index 0 is the embedding output.
- Judge files, followed by `asr.json` or `utility.json` after evaluation.

Direction estimation produces a `.npz` containing directions and calibration identities, plus a readable JSON sidecar. Use the `.npz` for inference and analysis. Model weights, datasets, directions, and hidden states are not bundled with the source code.
