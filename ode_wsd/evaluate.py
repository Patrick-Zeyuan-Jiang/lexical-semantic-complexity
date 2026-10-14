from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from .bank import chunks
from .common import digest, file_sha, read_json, read_jsonl, write_csv, write_json
from .dataset import verify_dataset


def wilson_interval(correct, total, z=1.96):
    if not total:
        return [None, None]
    p = correct / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    delta = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, centre - delta), min(1.0, centre + delta)]


def summarize(rows):
    n = len(rows)
    if not n:
        raise ValueError("没有可评测的标注样本。")
    correct = sum(bool(r["main_correct"]) for r in rows)
    covered = sum(r["status"] == "ok" for r in rows)
    by_gold = defaultdict(list)
    predicted, hits = Counter(), Counter()
    for r in rows:
        by_gold[r["gold_main_sense_id"]].append(r)
        if r.get("pred_main_sense_id"):
            predicted[r["pred_main_sense_id"]] += 1
        if r["main_correct"]:
            hits[r["gold_main_sense_id"]] += 1
    per_sense = []
    for sid in sorted(set(by_gold) | set(predicted)):
        support, tp = len(by_gold[sid]), hits[sid]
        precision = tp / predicted[sid] if predicted[sid] else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_sense.append(dict(main_sense_id=sid, support=support, predicted=predicted[sid],
                              correct=tp, precision=precision, recall=recall, f1=f1))
    by_entry = defaultdict(list)
    for r in rows:
        by_entry[r.get("gold_entry_key", "external")].append(r)
    per_entry = [{"entry_key": eid, "n": len(group),
                  "accuracy": sum(r["main_correct"] for r in group) / len(group)}
                 for eid, group in sorted(by_entry.items())]
    metrics = {"n_test": n, "correct_main": correct, "accuracy_main": correct / n,
               "coverage": covered / n, "n_uncovered": n - covered,
               "accuracy_main_on_covered": correct / covered if covered else None,
               "macro_recall_main": float(np.mean([s["recall"] for s in per_sense if s["support"]])),
               "macro_f1_main": float(np.mean([s["f1"] for s in per_sense])),
               "macro_accuracy_entry": float(np.mean([e["accuracy"] for e in per_entry])),
               "wilson_95_interval_main": wilson_interval(correct, n),
               "interval_note": "近似二项区间；同词条例句并非完全独立，不表示跨语域泛化区间。",
               "errors": dict(Counter(r["error_type"] for r in rows if not r["main_correct"]))}
    return metrics, per_sense, per_entry


def save_reports(out, rows, metadata):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    metrics, per_sense, per_entry = summarize(rows)
    metrics.update(metadata)
    write_csv(out / "predictions.csv", rows)
    write_csv(out / "errors.csv", [r for r in rows if not r["main_correct"]], fields=list(rows[0]))
    write_csv(out / "per_main_sense.csv", per_sense)
    write_csv(out / "per_entry.csv", per_entry)
    confusion = Counter((r["gold_main_sense_id"], r.get("pred_main_sense_id") or "<NO_PREDICTION>") for r in rows)
    write_csv(out / "confusion_pairs.csv", [{"gold": g, "predicted": p, "count": n}
                                          for (g, p), n in sorted(confusion.items())])
    write_json(out / "metrics.json", metrics)
    return metrics


def evaluate_holdout(dataset_dir, tool, output_dir, *, chunk_size=256):
    dataset = Path(dataset_dir)
    dm = verify_dataset(dataset)
    if dm["dataset_id"] != tool.manifest["dataset_id"]:
        raise ValueError("测试集与向量库的数据集 ID 不一致。")
    train_main_counts = Counter()
    for s in tool.senses:
        train_main_counts[s["main_sense_id"]] += s["n_train"]
    rows = []
    for batch in tqdm(chunks(read_jsonl(dataset / "test.jsonl.gz"), chunk_size),
                      total=math.ceil(dm["counts"]["test_examples"] / chunk_size), desc="评测留出集"):
        if any(r.get("split") != "test" for r in batch):
            raise ValueError("测试文件出现非测试行。")
        vectors = tool.encoder.encode(batch)
        for r, vector in zip(batch, vectors):
            # End-to-end candidate lookup does not read gold_entry_key/gold_pos.
            pred = tool.rank(r, vector)
            oracle = tool.rank(r, vector, restrict_entry=r["entry_key"])
            indices, _, _ = tool.candidates(r)
            main_candidates = {tool.senses[i]["main_sense_id"] for i in indices}
            majority = min(main_candidates, key=lambda s: (-train_main_counts[s], s)) if main_candidates else None
            first = min(main_candidates, key=lambda sid: (
                tool.sense_map[sid]["entry_key"], int(tool.sense_map[sid]["homograph"]),
                tool.sense_map[sid]["pos"], int(tool.sense_map[sid]["number"]))) if main_candidates else None
            correct = pred["main_sense_id"] == r["main_sense_id"]
            error = "correct" if correct else (
                "no_candidate" if pred["status"] != "ok" else
                "gold_missing_from_candidates" if r["main_sense_id"] not in main_candidates else "wrong_main_sense")
            rows.append({"instance_id": r["instance_id"], "text": r["text"],
                         "target": r["target_text"], "target_start": r["target_start"], "target_end": r["target_end"],
                         "gold_entry_key": r["entry_key"], "gold_pos": r["gold_pos"],
                         "gold_main_sense_id": r["main_sense_id"], "gold_fine_sense_id": r["sense_id"],
                         "pred_main_sense_id": pred["main_sense_id"], "pred_fine_sense_id": pred["fine_sense_id"],
                         "status": pred["status"], "main_correct": correct,
                         "fine_correct": pred["fine_sense_id"] == r["sense_id"],
                         "oracle_main_correct": oracle["main_sense_id"] == r["main_sense_id"],
                         "candidate_recall": r["main_sense_id"] in main_candidates,
                         "majority_correct": majority == r["main_sense_id"],
                         "first_sense_correct": first == r["main_sense_id"],
                         "random_expected_accuracy": 1 / len(main_candidates) if r["main_sense_id"] in main_candidates else 0.0,
                         "n_candidate_main": len(main_candidates), "cosine": pred["cosine"],
                         "main_margin": pred["main_margin"], "error_type": error})
    n = len(rows)
    if n != dm["counts"]["test_examples"]:
        raise ValueError("测试行数与数据清单不一致。")
    metrics = save_reports(output_dir, rows, {
        "dataset_id": dm["dataset_id"], "bank_fingerprint": tool.manifest["fingerprint"],
        "model": tool.manifest["encoder"], "sample_mode": dm["sample_mode"],
        "scope": "词典例句内部留出评测；不等于真实作文或通用 WSD 准确率。",
        "candidate_mode": "predicted_NLTK_lemma_and_surface_alias_union",
        "pos_filter": tool.pos_filter,
        "accuracy_fine": sum(r["fine_correct"] for r in rows) / n,
        "oracle_entry_accuracy_main": sum(r["oracle_main_correct"] for r in rows) / n,
        "candidate_recall_main": sum(r["candidate_recall"] for r in rows) / n,
        "baseline_training_majority_accuracy": sum(r["majority_correct"] for r in rows) / n,
        "baseline_first_sense_accuracy": sum(r["first_sense_correct"] for r in rows) / n,
        "baseline_random_expected_accuracy": sum(r["random_expected_accuracy"] for r in rows) / n,
        "baseline_note": "多数义基线只用训练例句数；词典例句数不等于自然语言义项频率。first 基线按词条/类别号/词性/主义项号排序。",
        "test_leakage": 0,
    })
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def make_annotation_template(tool, texts, output_csv, *, sample_size=300, seed=20260929):
    """Blind annotation sheet: no model prediction column and no guessed gold label."""
    import random
    rows = []
    for doc_id, text in texts.items():
        for a, b in tool.processor.sentence_spans(text):
            sentence = text[a:b]
            tokens = tool.processor.analyze(sentence)
            for t in tokens:
                request = tool._request(sentence, tokens, t)
                indices, _, _ = tool.candidates(request)
                if not indices:
                    continue
                ids = sorted({tool.senses[i]["main_sense_id"] for i in indices})
                definitions = {sid: tool.sense_map[sid]["definition"] for sid in ids}
                rows.append({"instance_id": digest([doc_id, a + t["start"], a + t["end"]]),
                             "document_id": doc_id, "text": sentence, "target": t["text"],
                             "target_start": t["start"], "target_end": t["end"],
                             "document_target_start": a + t["start"],
                             "candidate_main_definitions": definitions, "gold_main_sense_id": "",
                             "annotator_notes": ""})
    random.Random(seed).shuffle(rows)
    selected = rows[:sample_size]
    write_csv(output_csv, selected, fields=["instance_id", "document_id", "text", "target", "target_start", "target_end",
                                          "document_target_start", "candidate_main_definitions", "gold_main_sense_id", "annotator_notes"])
    return len(selected)


def evaluate_annotated_csv(tool, csv_path, output_dir):
    """Gold IDs separated by | are accepted. OUT_OF_INVENTORY is a valid gold label.

    Blank gold rows are counted, never treated as correct. A fully blank file
    raises instead of publishing a meaningless accuracy.
    """
    rows, unannotated = [], 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        records = list(csv.DictReader(f))
    seen = set()
    for r in tqdm(records, desc="人工标注集评测"):
        gold = [x.strip() for x in r.get("gold_main_sense_id", "").split("|") if x.strip()]
        if not gold:
            unannotated += 1
            continue
        for sid in gold:
            if sid != "OUT_OF_INVENTORY" and (sid not in tool.sense_map or tool.sense_map[sid]["level"] != 1):
                raise ValueError(f"人工标注含未知或非主义项 ID：{sid}")
        instance_id = r.get("instance_id") or digest([r["text"], r["target_start"], r["target_end"]])
        if instance_id in seen:
            raise ValueError(f"人工标注出现重复 instance_id：{instance_id}")
        seen.add(instance_id)
        pred = tool.predict_target(r["text"], r["target"], start=int(r["target_start"]), end=int(r["target_end"]))
        correct = pred["main_sense_id"] in gold
        # Select a matched valid alternative for single-label confusion summaries.
        scoring_gold = pred["main_sense_id"] if correct else gold[0]
        rows.append({"instance_id": instance_id, "text": r["text"], "target": r["target"],
                     "gold_entry_key": tool.sense_map.get(scoring_gold, {}).get("entry_key", "OUT_OF_INVENTORY"),
                     "gold_main_sense_id": scoring_gold, "accepted_gold_ids": gold,
                     "pred_main_sense_id": pred["main_sense_id"], "status": pred["status"],
                     "main_correct": correct, "cosine": pred["cosine"],
                     "error_type": "correct" if correct else "out_of_inventory" if "OUT_OF_INVENTORY" in gold else "wrong_or_uncovered"})
    return save_reports(output_dir, rows, {"scope": "人工标注外部语料；分数仅适用于该抽样范围。",
                                          "unannotated_rows": unannotated,
                                          "multi_gold_scoring": "任一可接受主义项命中算对；宏平均和混淆表使用命中项，否则使用第一标注项。",
                                          "annotations_sha256": file_sha(csv_path),
                                          "bank_fingerprint": tool.manifest["fingerprint"]})
