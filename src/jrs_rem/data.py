import os
import json
import csv
from pathlib import Path
from .io import atomic_json, validate_samples


# =========================
# Shared utility
# =========================
def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)


# =========================
# HADES
# =========================
def load_HADES(root, hades_step=4):
    DATA_PATH = os.path.join(root, "HADES")
    DATASET = "HADES"
    UIMG_TYPES = ["SD", "SD_typo", "typo"]

    all_data = []

    txt_type = "utxt"
    classes = ["Animal", "Financial", "Privacy", "Self-Harm", "Violence"]

    for data_class in classes:
        raw_data = _load_json(
            os.path.join(DATA_PATH, "instructions", f"{data_class}.json")
        )

        for item in raw_data:
            prompt_idx = item["id"]

            for img_tp in UIMG_TYPES:
                if img_tp == "SD":
                    img_path = os.path.join(
                        DATA_PATH, "optimized_SD_images",
                        data_class, str(prompt_idx), f"opt_{hades_step}.jpg"
                    )
                elif img_tp == "SD_typo":
                    img_path = os.path.join(
                        DATA_PATH, "optimized_SD_images",
                        data_class, str(prompt_idx), f"concat_opt_{hades_step}.jpg"
                    )
                elif img_tp == "typo":
                    img_path = os.path.join(
                        DATA_PATH, "optimized_SD_images",
                        data_class, str(prompt_idx), "keyword.jpg"
                    )

                _check_path(img_path)

                all_data.append({
                    "dataset": DATASET,
                    "idx": prompt_idx,
                    "class": data_class,
                    "data_type": f"{txt_type}+uimg[{img_tp}]",
                    "txt": item["instruction"],
                    "img": img_path,
                })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# MM-SafetyBench
# =========================
def load_MM_SafetyBench(root, hades_step=4):
    DATA_PATH = os.path.join(root, "MM-SafetyBench")
    DATASET = "MM-SafetyBench"
    UIMG_TYPES = ["SD", "SD_TYPO", "TYPO"]

    utxt_dir = os.path.join(DATA_PATH, "processed_questions")
    files = sorted(f for f in os.listdir(utxt_dir) if f.endswith(".json"))

    all_data = []

    for file_name in files:
        raw_data = _load_json(os.path.join(utxt_dir, file_name))
        data_class = file_name.replace(".json", "").replace(".txt", "")

        for prompt_idx, key in enumerate(raw_data):
            txt = raw_data[key]["Changed Question"]

            for img_tp in UIMG_TYPES:
                img_path = os.path.join(
                    DATA_PATH, "imgs", data_class, img_tp, f"{prompt_idx}.jpg"
                )

                _check_path(img_path)

                all_data.append({
                    "dataset": DATASET,
                    "idx": prompt_idx,
                    "class": data_class,
                    "data_type": f"utxt+uimg[{img_tp}]",
                    "txt": txt,
                    "img": img_path,
                })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# RedTeam_2K
# =========================
def load_RedTeam_2K(root, hades_step=4):
    DATA_PATH = os.path.join(root, "RedTeam_2K")
    DATASET = "RedTeam_2K"
    UIMG_TYPES = ["SD", "SD_typo", "typo"]

    with open(os.path.join(DATA_PATH, "RedTeam_2K_filtered.csv"), newline="", encoding="utf-8") as f:
        df = list(csv.DictReader(f))

    all_data = []

    for row in df:
        prompt_idx = row["id"]
        data_class = row["policy"]

        for img_tp in UIMG_TYPES:
            img_path = os.path.join(
                DATA_PATH, "imgs", f"{img_tp}_related_{prompt_idx}.png"
            )

            _check_path(img_path)

            all_data.append({
                "dataset": DATASET,
                "idx": prompt_idx,
                "class": data_class,
                "data_type": f"utxt+uimg[{img_tp}]",
                "txt": row["question"],
                "img": img_path,
            })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# HADES hideharm
# =========================
def load_HADES_hideharm(root, hades_step=4):
    DATA_PATH = os.path.join(root, "HADES")
    DATASET = "HADES_hideharm"

    classes = ["Animal", "Financial", "Privacy", "Self-Harm", "Violence"]

    all_data = []

    for data_class in classes:
        mml = _load_json(os.path.join(DATA_PATH, "MML", data_class, "data.json"))
        raw = _load_json(os.path.join(DATA_PATH, "instructions", f"{data_class}.json"))

        if len(mml) != len(raw):
            raise ValueError(f"HADES implicit/explicit sample count mismatch: {data_class}")

        for r, m in zip(raw, mml):
            if r["instruction"] != m["instruction"]:
                raise ValueError(f"HADES implicit/explicit instruction mismatch: {data_class}")

            prompt_idx = r["id"]

            img_path = os.path.join(
                DATA_PATH,
                "optimized_SD_images",
                data_class,
                str(prompt_idx),
                f"concat_opt_{hades_step}.jpg",
            )

            _check_path(img_path)

            all_data.append({
                "dataset": DATASET,
                "idx": prompt_idx,
                "class": data_class,
                "data_type": "m_utxt+uimg[SD_typo]",
                "txt": m["hades_prompt"],
                "img": img_path,
            })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# MM-SafetyBench hideharm
# =========================
def load_MM_SafetyBench_hideharm(root, hades_step=4):
    DATA_PATH = os.path.join(root, "MM-SafetyBench")
    DATASET = "MM_SafetyBench_hideharm"
    UIMG_TYPES = ["SD_TYPO"]

    utxt_dir = os.path.join(DATA_PATH, "processed_questions")
    files = sorted(f for f in os.listdir(utxt_dir) if f.endswith(".json"))

    all_data = []

    for file_name in files:
        raw = _load_json(os.path.join(utxt_dir, file_name))
        data_class = file_name.replace(".json", "")

        for idx, key in enumerate(raw):
            for img_tp in UIMG_TYPES:
                img_path = os.path.join(
                    DATA_PATH, "imgs", data_class, img_tp, f"{idx}.jpg"
                )

                _check_path(img_path)

                txt = raw[key].get("Rephrased Question(SD)")
                if img_tp == "SD_TYPO":
                    txt = raw[key].get("Rephrased Question")

                all_data.append({
                    "dataset": DATASET,
                    "idx": idx,
                    "class": data_class,
                    "data_type": f"m_utxt+uimg[{img_tp.lower()}]",
                    "txt": txt,
                    "img": img_path,
                })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# MMVet
# =========================
def load_MMvet(root, hades_step=4):
    DATA_PATH = os.path.join(root, "MMvet")
    DATASET = "MMvet"

    raw = _load_json(os.path.join(DATA_PATH, "mm-vet.json"))

    all_data = []

    for k, v in raw.items():
        img_path = os.path.join(DATA_PATH, "images", v["imagename"])
        _check_path(img_path)

        all_data.append({
            "dataset": DATASET,
            "idx": k,
            "class": DATASET,
            "data_type": "stxt+rimg",
            "txt": v["question"],
            "img": img_path,
        })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# MME
# =========================
def load_MME(root, hades_step=4):
    DATA_PATH = os.path.join(root, "MME")
    DATASET = "MME"

    raw = _load_json(os.path.join(DATA_PATH, "MME.json"))

    all_data = []

    for v in raw:
        img_path = os.path.join(DATA_PATH, v["img"])
        _check_path(img_path)

        all_data.append({
            "dataset": DATASET,
            "idx": v["question_id"],
            "class": v["category"],
            "data_type": "stxt+rimg",
            "txt": v["question"],
            "img": img_path,
        })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# ScienceQA
# =========================
def load_ScienceQA(root, hades_step=4):
    DATA_PATH = os.path.join(root, "ScienceQA")
    DATASET = "ScienceQA"

    raw = _load_json(os.path.join(DATA_PATH, "test_data.json"))

    all_data = []

    for v in raw:
        if v["img"] is None:
            all_data.append({
                "dataset": DATASET,
                "idx": v["idx"],
                "class": v["cls"],
                "data_type": "stxt",
                "txt": v["question"],
                "img": None,
            })
        else:
            img_path = os.path.join(DATA_PATH, v["img"])
            _check_path(img_path)

            all_data.append({
                "dataset": DATASET,
                "idx": v["idx"],
                "class": v["cls"],
                "data_type": "stxt+rimg",
                "txt": v["question"],
                "img": img_path,
            })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# RealWorldQA
# =========================
def load_RealWorldQA(root, hades_step=4):
    DATA_PATH = os.path.join(root, "RealWorldQA")
    DATASET = "RealWorldQA"

    raw = _load_json(os.path.join(DATA_PATH, "realworldqa.json"))

    all_data = []

    for v in raw:
        img_path = os.path.join(DATA_PATH, "images", f"{v['idx']:04d}.png")
        _check_path(img_path)

        all_data.append({
            "dataset": DATASET,
            "idx": v["idx"],
            "class": DATASET,
            "data_type": "stxt+rimg",
            "txt": v["question"],
            "img": img_path,
        })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# MMMU (validation, 仅单图题)
# =========================
def load_MMMU(root, hades_step=4):
    DATA_PATH = os.path.join(root, "MMMU")
    DATASET = "MMMU"

    raw = _load_json(os.path.join(DATA_PATH, "mmmu_val.json"))

    all_data = []

    for v in raw:
        img_path = os.path.join(DATA_PATH, "images", f"{v['idx']}.png")
        _check_path(img_path)

        all_data.append({
            "dataset": DATASET,
            "idx": v["idx"],
            "class": v["subject"],
            "data_type": "stxt+rimg",
            "txt": v["question"],
            "img": img_path,
        })

    print(f"{DATASET} loaded. Total {len(all_data)} samples.")
    return all_data


# =========================
# Dispatcher
# =========================
def load_dataset(name, root, hades_step=4):
    mapping = {
        "HADES": load_HADES,
        "MM-SafetyBench": load_MM_SafetyBench,
        "RedTeam_2K": load_RedTeam_2K,
        "HADES_hideharm": load_HADES_hideharm,
        "MM_SafetyBench_hideharm": load_MM_SafetyBench_hideharm,
        "MMvet": load_MMvet,
        "MME": load_MME,
        "ScienceQA": load_ScienceQA,
        "RealWorldQA": load_RealWorldQA,
        "MMMU": load_MMMU,
    }

    if name not in mapping:
        raise ValueError(f"Unknown dataset: {name}")

    return validate_samples(mapping[name](str(Path(root).resolve()), hades_step))

def prepare(name, root, output, hades_step=4):
    rows = load_dataset(name, root, hades_step)
    atomic_json(output, rows)
    return rows
