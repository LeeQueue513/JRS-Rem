# JRS-Rem

Code for **[Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift](https://openreview.net/forum?id=ovIB52cIlw)**, NeurIPS 2026.

JRS-Rem is an inference-time defense for vision-language models. It estimates a jailbreak direction from refusal and jailbreak representations, then removes the image-induced shift along that direction. The target model's parameters remain frozen.

This repository provides the **LLaVA-1.5-7B** implementation, safety and utility evaluation, and representation analyses: PCA, cosine distances, linear probes, safety warnings, and image-induced shifts.

## Installation

Use Python 3.10 or later and a PyTorch installation suitable for your device. From the repository root:

```bash
python -m pip install -e '.[inference,analysis]'
jrs-rem --help
```

Obtain `llava-hf/llava-1.5-7b-hf`, the judge checkpoints, and datasets separately. See [data preparation](docs/data.md) for supported layouts and custom inputs.

## Quick start

Run commands from the repository root. Edit the configs' model, device, and paths for your environment. Config paths are relative to the config file; local relative model paths must start with `./` or `../`.

### 1. Collect baseline answers and representations

```bash
jrs-rem prepare --dataset HADES --data-root /path/to/datasets --hades-step 4 --output data/hades.json
jrs-rem infer --config configs/baseline.json
```

### 2. Judge answers and estimate the direction

```bash
jrs-rem judge --results outputs/baseline --judge machine
jrs-rem judge --results outputs/baseline --judge Qwen3Guard-Gen-8B --model /path/to/Qwen3Guard-Gen-8B
jrs-rem judge --results outputs/baseline --judge Llama-Guard-4-12B --model /path/to/Llama-Guard-4-12B
jrs-rem direction --results outputs/baseline --count 50 --output outputs/direction.npz
```

Calibration selects 50 refusal and 50 jailbreak prompt groups with unanimous labels, using one image variant per prompt and disjoint groups. Each layer's direction is the normalized difference between the two mean representations.

### 3. Compare baseline and defense

```bash
jrs-rem infer --config configs/baseline_heldout.json
jrs-rem infer --config configs/defense.json
```

Both configs use the same direction file to exclude all calibration prompt variants, including implicit variants. Repeat the three judge commands for each output directory, then report ASR:

```bash
jrs-rem asr --results outputs/baseline_heldout
jrs-rem asr --results outputs/defense
```

## Further usage

- [Data preparation and outputs](docs/data.md)
- [Experiment settings, evaluation, and analysis commands](docs/reproduction.md)

Run tests with:

```bash
python -m unittest discover -s tests -v
```

## Citation

```bibtex
@inproceedings{wei2026jrsrem,
  title={Understanding and Defending VLM Jailbreaks via Jailbreak-Related Representation Shift},
  author={Wei, Zhihua and Li, Qiang and Ruan, Jian and Qin, Zhenxin and Wen, Leilei and Qin, Ruiyang and Wang, Qingzhuo and Liu, Dongrui and Shen, Wen},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026},
  url={https://openreview.net/forum?id=ovIB52cIlw}
}
```

## License

The source code and accompanying documentation are released under the [MIT License](LICENSE). Model checkpoints and datasets are governed by their respective licenses.
