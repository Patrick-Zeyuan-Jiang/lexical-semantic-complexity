"""Streaming HTML ingestion, NLTK target alignment, global de-duplication, holdout."""
from __future__ import annotations

import gzip
import json
import math
import random
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

from tqdm.auto import tqdm

from .common import (VERSION, DataConfig, config_dict, digest, file_sha, normalize,
                     read_json, read_jsonl, sentence_key, validate_export,
                     write_json, write_jsonl, implementation_digest)
from .parser import parse_record
from .nlp import NLTKProcessor, locate_example


def merge_unique(left, right):
    values = {digest(x): x for x in left}
    for x in right:
        values.setdefault(digest(x), x)
    return list(values.values())


# ODE_WSD_CONFLICT_QUARANTINE_V1
def add_record(db, record, config, counts, *, conflict_path=None):
    entries, senses, stats = parse_record(record, config)
    counts.update(stats)
    entry_ids = {e["entry_key"] for e in entries} | {s["entry_key"] for s in senses}
    blocked = {eid for eid in entry_ids if db.execute(
        "SELECT 1 FROM quarantined_entries WHERE id=?", (eid,)).fetchone()}

    # Preflight all senses, including repeats within this very MDX record.
    pending = {}
    for s in senses:
        if s["entry_key"] in blocked:
            continue
        sid = s["sense_id"]
        old = pending.get(sid)
        if old is None:
            prior = db.execute("SELECT payload FROM senses WHERE id=?", (sid,)).fetchone()
            old = json.loads(prior[0]) if prior else None
        if old is not None and normalize(old["definition"]) != normalize(s["definition"]):
            if conflict_path is None:
                raise RuntimeError("冲突处理必须提供 conflict_path，以保留诊断信息。")
            affected = sorted({old["entry_key"], s["entry_key"]})
            event = {"reason": "same_sense_id_different_definition",
                     "action": "exclude_entire_entry", "sense_id": sid,
                     "excluded_entries": affected,
                     "processed_records": counts.get("source_records", 0),
                     "old_sense": old, "new_sense": s}
            # Close on each event so the evidence survives a later parse failure.
            with Path(conflict_path).open("a", encoding="utf-8") as log:
                log.write(json.dumps(event, ensure_ascii=False) + "\n")
            for eid in affected:
                db.execute("INSERT OR IGNORE INTO quarantined_entries VALUES (?)", (eid,))
                removed = db.execute("SELECT COUNT(*) FROM senses WHERE entry_id=?", (eid,)).fetchone()[0]
                counts["previous_senses_removed_due_to_conflict"] += removed
                db.execute("DELETE FROM senses WHERE entry_id=?", (eid,))
                db.execute("DELETE FROM entries WHERE id=?", (eid,))
                blocked.add(eid)
            counts["sense_definition_conflicts"] += 1
            counts["excluded_conflicting_entries"] = db.execute(
                "SELECT COUNT(*) FROM quarantined_entries").fetchone()[0]
        pending[sid] = s

    # The quarantine persists for the entire input pass, including later records.
    for e in entries:
        if e["entry_key"] in blocked:
            continue
        prior = db.execute("SELECT payload FROM entries WHERE id=?", (e["entry_key"],)).fetchone()
        if prior:
            old = json.loads(prior[0])
            e["forms"] = merge_unique(old["forms"], e["forms"])
        db.execute("INSERT OR REPLACE INTO entries VALUES (?,?)",
                   (e["entry_key"], json.dumps(e, ensure_ascii=False)))
    for s in senses:
        if s["entry_key"] in blocked:
            counts["senses_skipped_due_to_entry_conflict"] += 1
            continue
        sid = s["sense_id"]
        prior = db.execute("SELECT payload FROM senses WHERE id=?", (sid,)).fetchone()
        if prior:
            old = json.loads(prior[0])
            # Keep the original invariant; unexpected logic errors must still stop.
            if normalize(old["definition"]) != normalize(s["definition"]):
                raise ValueError(f"同一义项 ID 对应不同定义，拒绝覆盖：{sid}")
            examples = {sentence_key(e["text"]): e for e in old["examples"]}
            for ex in s["examples"]:
                examples.setdefault(sentence_key(ex["text"]), ex)
            s["examples"] = list(examples.values())
            s["sources"] = merge_unique(old["sources"], s["sources"])
            counts["duplicate_sense_records_merged"] += 1
        db.execute("INSERT OR REPLACE INTO senses VALUES (?,?,?,?,?,?,?,?)",
                   (sid, s["entry_key"], s["level"], s["main_sense_id"],
                    len(s["examples"]), 0, 0, json.dumps(s, ensure_ascii=False)))


def prepare_dataset(input_path, output_dir, config=None, *, processor=None,
                    sample_mode=False, work_dir=None):
    config = config or DataConfig()
    config.validate()
    expected = validate_export(input_path, sample_mode)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Source identity covers bytes, not just a filename/time; test fixtures are explicit.
    import nltk
    signature = {"software_version": VERSION, "source_sha256": file_sha(input_path),
                 "implementation_sha256": implementation_digest(),
                 "config": config_dict(config), "sample_mode": sample_mode,
                 "definition_conflict_policy": "exclude_entire_entry_and_log",
                 "nlp_version": nltk.__version__,
                 "nlp_processor": type(processor).__name__ if processor is not None else "NLTKProcessor"}
    signature["fingerprint"] = digest(signature)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        old = read_json(manifest_path)
        if old["fingerprint"] != signature["fingerprint"]:
            raise ValueError("该数据目录已有不同输入或配置。请修改 RUN_NAME，保留原实验。")
        verify_dataset(out)
        print("复用已完成的数据集：", out)
        return old
    processor = processor or NLTKProcessor()
    counts = Counter()
    with tempfile.TemporaryDirectory(prefix="ode_parse_", dir=work_dir) as scratch:
        db = sqlite3.connect(str(Path(scratch) / "catalog.sqlite"))
        db.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE entries (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE quarantined_entries (id TEXT PRIMARY KEY);
            CREATE TABLE senses (id TEXT PRIMARY KEY, entry_id TEXT, level INTEGER,
                main_id TEXT, n_raw INTEGER, n_valid INTEGER, keep INTEGER, payload TEXT);
            CREATE INDEX senses_entry ON senses(entry_id);
            CREATE TABLE examples (id TEXT PRIMARY KEY, sid TEXT, text_hash TEXT,
                split TEXT, payload TEXT);
            CREATE INDEX ex_sid ON examples(sid);
            CREATE INDEX ex_text ON examples(text_hash);
        """)
        conflict_path = out / "sense_conflicts.jsonl"
        conflict_path.write_text("", encoding="utf-8")
        counts["sense_definition_conflicts"] = 0
        counts["excluded_conflicting_entries"] = 0
        try:
            for row in tqdm(read_jsonl(input_path), total=expected, desc="解析 GB 词典", unit="条"):
                counts["source_records"] += 1
                try:
                    add_record(db, row, config, counts, conflict_path=conflict_path)
                except Exception as e:
                    write_json(out / "parse_failure.json", {
                        "record_id": row.get("entry_id"), "headword": row.get("headword"),
                        "error": repr(e), "processed_records": counts["source_records"]})
                    raise
                if counts["source_records"] % 2000 == 0:
                    db.commit()
            db.commit()
            if expected is not None and counts["source_records"] != expected:
                raise ValueError(f"记录数不一致：读取 {counts['source_records']}，应为 {expected}")
            counts["unique_entries_before_filter"] = db.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
            counts["unique_senses_before_filter"] = db.execute("SELECT COUNT(*) FROM senses").fetchone()[0]
            candidates = [r[0] for r in db.execute(
                "SELECT entry_id FROM senses WHERE level=1 AND n_raw>=? GROUP BY entry_id HAVING COUNT(*)>=? ORDER BY entry_id",
                (config.min_raw_examples, config.min_main_senses))]
            counts["entries_passing_raw_filter"] = len(candidates)
            rejection_path = out / "example_rejections.jsonl.gz"
            with gzip.open(rejection_path, "wt", encoding="utf-8") as rejection_log:
                def reject(row):
                    counts["rejected_" + row["reason"]] += 1
                    rejection_log.write(json.dumps(row, ensure_ascii=False) + "\n")

                for eid in tqdm(candidates, desc="NLTK 定位目标词", unit="词"):
                    entry = json.loads(db.execute("SELECT payload FROM entries WHERE id=?", (eid,)).fetchone()[0])
                    values = [json.loads(r[0]) for r in db.execute(
                        "SELECT payload FROM senses WHERE entry_id=? AND n_raw>=? ORDER BY id",
                        (eid, config.min_raw_examples))]
                    legal_main = {s["sense_id"] for s in values if s["level"] == 1}
                    for sense in values:
                        if sense["main_sense_id"] not in legal_main:
                            continue
                        for ex in sense["examples"]:
                            located, reason = locate_example(ex["text"], entry, processor)
                            if reason:
                                reject({"sense_id": sense["sense_id"], "text": ex["text"], "reason": reason})
                                continue
                            located.update({"sense_id": sense["sense_id"], "main_sense_id": sense["main_sense_id"],
                                            "entry_key": eid, "gold_pos": sense["pos"],
                                            "source_type": ex["source_type"]})
                            text_hash = digest(sentence_key(located["text"]))
                            located["instance_id"] = digest([sense["sense_id"], text_hash,
                                                              located["target_start"], located["target_end"]])
                            db.execute("INSERT OR IGNORE INTO examples VALUES (?,?,?,?,?)",
                                       (located["instance_id"], sense["sense_id"], text_hash, "",
                                        json.dumps(located, ensure_ascii=False)))
                    db.commit()
                # Remove shared contexts globally, including parent/child and different
                # entries, before random splitting. Exact quota per retained sense is preserved.
                db.execute("CREATE TEMP TABLE repeated AS SELECT text_hash FROM examples GROUP BY text_hash HAVING COUNT(DISTINCT sid)>1")
                for (payload,) in db.execute("SELECT payload FROM examples WHERE text_hash IN (SELECT text_hash FROM repeated)"):
                    r = json.loads(payload)
                    reject({"sense_id": r["sense_id"], "text": r["text"], "reason": "cross_sense_duplicate"})
                db.execute("DELETE FROM examples WHERE text_hash IN (SELECT text_hash FROM repeated)")
            db.execute("UPDATE senses SET n_valid=(SELECT COUNT(*) FROM examples WHERE examples.sid=senses.id)")
            db.execute("UPDATE senses SET keep=1 WHERE level=1 AND n_raw>=? AND n_valid>=?",
                       (config.min_raw_examples, config.min_usable_examples))
            db.execute("UPDATE senses SET keep=0 WHERE entry_id NOT IN (SELECT entry_id FROM senses WHERE keep=1 GROUP BY entry_id HAVING COUNT(*)>=?)",
                       (config.min_main_senses,))
            db.execute("UPDATE senses SET keep=1 WHERE level=2 AND n_raw>=? AND n_valid>=? AND main_id IN (SELECT id FROM senses WHERE level=1 AND keep=1)",
                       (config.min_raw_examples, config.min_usable_examples))
            db.execute("DELETE FROM examples WHERE sid NOT IN (SELECT id FROM senses WHERE keep=1)")
            db.commit()
            counts["retained_main_senses"] = db.execute("SELECT COUNT(*) FROM senses WHERE keep=1 AND level=1").fetchone()[0]
            counts["retained_subsenses"] = db.execute("SELECT COUNT(*) FROM senses WHERE keep=1 AND level=2").fetchone()[0]
            if not counts["retained_main_senses"]:
                write_json(out / "summary.json", dict(counts))
                raise ValueError("筛选后没有可用义项；请检查输入与 example_rejections.jsonl.gz。")
            for sid, n in list(db.execute("SELECT id,n_valid FROM senses WHERE keep=1 ORDER BY id")):
                ids = [x[0] for x in db.execute("SELECT id FROM examples WHERE sid=? ORDER BY id", (sid,))]
                rng = random.Random(int(digest([config.seed, sid]), 16))
                n_test = max(1, math.floor(n * config.test_fraction))
                n_test = min(n_test, n - 1)
                selected = set(rng.sample(ids, n_test))
                db.executemany("UPDATE examples SET split=? WHERE id=?",
                               [("test" if x in selected else "train", x) for x in ids])
            db.commit()
            for split in ["train", "test"]:
                counts[split + "_examples"] = db.execute("SELECT COUNT(*) FROM examples WHERE split=?", (split,)).fetchone()[0]
                write_jsonl(out / (split + ".jsonl.gz"),
                            (dict(json.loads(r[0]), split=split) for r in db.execute(
                                "SELECT payload FROM examples WHERE split=? ORDER BY sid,id", (split,))))

            def sense_rows():
                for payload, n_raw, n_valid in db.execute("SELECT payload,n_raw,n_valid FROM senses WHERE keep=1 ORDER BY id"):
                    s = json.loads(payload)
                    s.pop("examples")
                    n_test = min(max(1, math.floor(n_valid * config.test_fraction)), n_valid - 1)
                    s.update(n_raw=n_raw, n_usable=n_valid, n_train=n_valid - n_test, n_test=n_test)
                    yield s

            write_jsonl(out / "senses.jsonl", sense_rows())
            kept_entries = [r[0] for r in db.execute("SELECT DISTINCT entry_id FROM senses WHERE keep=1 ORDER BY entry_id")]
            counts["retained_entries"] = len(kept_entries)
            write_jsonl(out / "entries.jsonl", (json.loads(db.execute(
                "SELECT payload FROM entries WHERE id=?", (eid,)).fetchone()[0]) for eid in kept_entries))

            def audit_rows():
                for sid, eid, level, main, n_raw, n_valid, keep in db.execute(
                    "SELECT id,entry_id,level,main_id,n_raw,n_valid,keep FROM senses ORDER BY id"):
                    reason = "retained" if keep else (
                        "too_few_raw_examples" if n_raw < config.min_raw_examples else
                        "too_few_usable_examples_or_not_preselected" if n_valid < config.min_usable_examples else
                        "entry_or_parent_not_eligible")
                    yield dict(sense_id=sid, entry_key=eid, level=level, main_sense_id=main,
                               n_raw=n_raw, n_usable=n_valid, retained=bool(keep), reason=reason)
            write_jsonl(out / "sense_audit.jsonl.gz", audit_rows())
        finally:
            db.close()
    import nltk
    signature.update({"input_path": str(input_path), "counts": dict(counts),
                      "nlp_version": nltk.__version__,
                      "nlp_processor": type(processor).__name__,
                      "source_expected_records": expected,
                      "evaluation_scope": "dictionary_example_holdout",
                      "deduplication": "casefold_NFC_whitespace_exact_sentence; drop_all_cross_sense_copies"})
    alias = {}
    for e in read_jsonl(out / "entries.jsonl"):
        for f in e["forms"]:
            alias.setdefault(f["key"], set()).add(e["entry_key"])
    collisions = {k: sorted(v) for k, v in alias.items() if len(v) > 1}
    write_json(out / "alias_collisions.json", collisions)
    signature["alias_collision_count"] = len(collisions)
    write_json(out / "summary.json", signature)
    filenames = ["sense_conflicts.jsonl", "entries.jsonl", "senses.jsonl", "train.jsonl.gz", "test.jsonl.gz",
                 "sense_audit.jsonl.gz", "example_rejections.jsonl.gz", "alias_collisions.json"]
    signature["files"] = {name: file_sha(out / name) for name in filenames}
    signature["dataset_id"] = digest({"signature": signature["fingerprint"], "files": signature["files"]})
    write_json(manifest_path, signature)  # Commit marker is always written last.
    (out / "parse_failure.json").unlink(missing_ok=True)
    if counts["excluded_conflicting_entries"]:
        print("因定义冲突排除的词条数：", counts["excluded_conflicting_entries"],
              "；请检查：", out / "sense_conflicts.jsonl")
    print(json.dumps(dict(counts), ensure_ascii=False, indent=2))
    return signature


def verify_dataset(path):
    p = Path(path)
    m = read_json(p / "manifest.json")
    for name, expected in m["files"].items():
        if not (p / name).is_file() or file_sha(p / name) != expected:
            raise ValueError(f"数据文件缺失或被修改：{name}。请使用新的实验目录重建。")
    # No huge in-memory vectors here: keep only normalized text hashes for audit.
    train = {digest(sentence_key(r["text"])) for r in read_jsonl(p / "train.jsonl.gz")}
    leaks = sum(digest(sentence_key(r["text"])) in train for r in read_jsonl(p / "test.jsonl.gz"))
    if leaks:
        raise ValueError(f"训练/测试上下文泄漏：{leaks}")
    return m
