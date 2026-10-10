import json
from pathlib import Path
import re
from .io import read_json



def parse_choice(text, n_choices):
    valid = "ABCDEFGHI"[:n_choices]
    m = re.search(rf"answer is[:\s]*\(?([{valid}])\b", text, re.IGNORECASE)
    if not m:
        m = re.search(rf"\(?\b([{valid}])\b\)?", text)
    return m.group(1).upper() if m else None


def normalize_short(text):
    text = re.sub(r"\s+", " ", str(text).strip().lower())
    return text.rstrip(".")


def match_short(pred_text, gt):
    gt_n = normalize_short(gt)
    out_n = normalize_short(pred_text)
    if out_n == gt_n:
        return True
    return bool(re.search(rf"\b{re.escape(gt_n)}\b", out_n))


# MME: per-category accuracy plus paired-image accuracy.
# 每类 score = acc*100 + acc+*100 (acc+ = 同图两题全对的比例), 类满分 200;
# Perception = 10 类之和 (满分 2000), Cognition = 4 类之和 (满分 800)。
MME_PERCEPTION = ["existence", "count", "position", "color", "posters",
                  "celebrity", "scene", "landmark", "artwork", "OCR"]
MME_COGNITION = ["commonsense_reasoning", "numerical_calculation",
                 "text_translation", "code_reasoning"]


def _judge_yes_no_start(pred_ans, true_ans):
    r = pred_ans.lower().strip()
    if r.startswith("no"):
        r = "no"
    elif r.startswith("yes"):
        r = "yes"
    else:
        return False, True  # 未解析
    return r == true_ans.lower(), False


def eval_mme(results, data_root):
    gt = {v["question_id"]: v for v in read_json(Path(data_root) / "MME/MME.json")}
    pairs = {}
    n_unparsed = 0
    for s in results:
        v = gt.get(s["idx"])
        if v is None:
            raise ValueError(f"Missing ground truth for {s['idx']}")
        ok, unparsed = _judge_yes_no_start(s["output_txt"], v["answer"])
        n_unparsed += unparsed
        pairs.setdefault((s["class"], s["img"]), []).append(ok)

    per_cat = {}
    for (cat, img), oks in pairs.items():
        if len(oks) != 2:
            raise ValueError(f"MME requires two questions per image: {img} has {len(oks)}")
        c = per_cat.setdefault(cat, [0, 0, 0])
        c[0] += sum(oks)      # correct
        c[1] += all(oks)      # img_correct (两题全对)
        c[2] += len(oks)      # total

    scores = {}
    for cat, (correct, img_correct, total) in per_cat.items():
        acc = correct / total * 100
        img_acc = img_correct / (total / 2) * 100
        scores[cat] = acc + img_acc

    perception = sum(scores[c] for c in MME_PERCEPTION if c in scores)
    cognition = sum(scores[c] for c in MME_COGNITION if c in scores)
    return perception, cognition, scores, n_unparsed


def parse_scienceqa_answer(pred_text, choices):
    """ScienceQA parsing order from experiments/eval_scienceQA.ipynb."""
    options = ["A", "B", "C", "D", "E"]
    answer = "FAILED"
    if pred_text in options:
        answer = pred_text
    elif len(pred_text) >= 2 and pred_text[0] in options and not pred_text[1].isalpha():
        answer = pred_text[0]
    elif (len(pred_text) >= 3 and not pred_text[0].isalpha()
          and pred_text[1] in options and not pred_text[2].isalpha()):
        answer = pred_text[1]
    else:
        for choice in choices:
            if choice.strip().lower() in pred_text.lower():
                answer = options[choices.index(choice)]
                break
    if answer == "FAILED":
        match = re.search(r"\b([A-E])\b", pred_text)
        if match and len(match.groups()) == 1:
            answer = match.groups()[0]
    return options.index(answer) if answer in options[:len(choices)] else -1


def eval_scienceqa(results, data_root):
    n_ok = n_tot = 0
    per_split = {"stxt": [0, 0], "stxt+rimg": [0, 0]}
    # test_data.json 提供 choices/answer, 按 idx 对齐
    gt = {str(v["idx"]): v for v in read_json(Path(data_root) / "ScienceQA/test_data.json")}
    for s in results:
        v = gt.get(str(s["idx"]))
        if v is None:
            raise ValueError(f"Missing ground truth for {s['idx']}")
        pred = parse_scienceqa_answer(s["output_txt"], v["choices"])
        ok = pred == v["answer"]
        n_ok += ok
        n_tot += 1
        sp = per_split[s["data_type"]]
        sp[0] += ok
        sp[1] += 1
    splits = {k: a / n for k, (a, n) in per_split.items() if n}
    return n_ok / n_tot, splits


def _count_options(question):
    letters = re.findall(r"^\s*([A-I])[.)]\s", question, re.MULTILINE)
    return max((ord(c) - 64 for c in letters), default=4)


def eval_realworldqa(results, data_root):
    gt = {v["idx"]: v for v in read_json(Path(data_root) / "RealWorldQA/realworldqa.json")}
    per_split = {"letter": [0, 0], "open": [0, 0]}
    n_unparsed = 0
    for s in results:
        v = gt.get(s["idx"])
        if v is None:
            raise ValueError(f"Missing ground truth for {s['idx']}")
        ans = v["answer"]
        if re.fullmatch(r"[A-Z]", ans):
            pred = parse_choice(s["output_txt"], _count_options(v["question"]))
            if pred is None:
                n_unparsed += 1
            ok = pred == ans
            sp = per_split["letter"]
        else:
            ok = match_short(s["output_txt"], ans)
            sp = per_split["open"]
        sp[0] += ok
        sp[1] += 1
    splits = {k: a / n for k, (a, n) in per_split.items() if n}
    overall = sum(a for a, _ in per_split.values()) / sum(n for _, n in per_split.values())
    return overall, splits, n_unparsed


def eval_mmmu(results, data_root):
    gt = {v["idx"]: v for v in read_json(Path(data_root) / "MMMU/mmmu_val.json")}
    per_subj = {}
    per_type = {"multiple-choice": [0, 0], "open": [0, 0]}
    n_unparsed = 0
    for s in results:
        v = gt.get(s["idx"])
        if v is None:
            raise ValueError(f"Missing ground truth for {s['idx']}")
        if v["qtype"] == "multiple-choice":
            pred = parse_choice(s["output_txt"], len(v["options"]))
            if pred is None:
                n_unparsed += 1
            ok = pred == v["answer"]
        else:
            ok = match_short(s["output_txt"], v["answer"])
        per_subj.setdefault(v["subject"], [0, 0])
        per_subj[v["subject"]][0] += ok
        per_subj[v["subject"]][1] += 1
        tp = per_type[v["qtype"]]
        tp[0] += ok
        tp[1] += 1
    subs = {c: a / n for c, (a, n) in sorted(per_subj.items())}
    types = {k: a / n for k, (a, n) in per_type.items() if n}
    overall = sum(a for a, _ in per_subj.values()) / sum(n for _, n in per_subj.values())
    return overall, subs, types, n_unparsed
