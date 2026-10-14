from __future__ import annotations

import itertools
import os
import shutil
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm

from .common import (VERSION, EncoderConfig, digest, file_sha, read_json, read_jsonl,
                     write_json, implementation_digest)
from .dataset import verify_dataset
from .encoder import ContextEncoder


def chunks(iterator, size):
    it = iter(iterator)
    while True:
        chunk = list(itertools.islice(it, size))
        if not chunk:
            break
        yield chunk


def save_npz(path, **data):
    p = Path(path)
    temp = p.with_name(p.name + ".partial")
    with temp.open("wb") as f:
        np.savez_compressed(f, **data)
    os.replace(temp, p)


def save_npy(path, value):
    p = Path(path)
    temp = p.with_name(p.name + ".partial")
    with temp.open("wb") as f:
        np.save(f, value, allow_pickle=False)
    os.replace(temp, p)


def build_bank(dataset_dir, bank_dir, encoder=None, config=None, *, chunk_size=512):
    if chunk_size < 1:
        raise ValueError("chunk_size 必须为正。")
    dataset, out = Path(dataset_dir), Path(bank_dir)
    dm = verify_dataset(dataset)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "checkpoints"
    ckpt.mkdir(exist_ok=True)
    lock_path = out / "encoder_lock.json"
    config = config or EncoderConfig()
    if encoder is None:
        if lock_path.exists():
            old_lock = read_json(lock_path)
            identity = old_lock["encoder"]
            for k in ["model_name", "layer", "max_length", "batch_size"]:
                if getattr(config, k) != identity[k]:
                    raise ValueError(f"编码配置 {k} 与现有检查点不同，请更改 RUN_NAME。")
            if config.revision not in ["main", identity["revision"]]:
                raise ValueError("revision 与现有检查点不同。")
            config = EncoderConfig(**{k: identity[k] for k in EncoderConfig.__dataclass_fields__})
        encoder = ContextEncoder(config)
    signature = {"software_version": VERSION, "dataset_id": dm["dataset_id"],
                 "implementation_sha256": implementation_digest(),
                 "encoder": encoder.identity, "chunk_size": chunk_size}
    signature["fingerprint"] = digest(signature)
    if lock_path.exists() and read_json(lock_path)["fingerprint"] != signature["fingerprint"]:
        raise ValueError("检查点的数据、模型或运行版本不一致；请使用新的实验目录。")
    write_json(lock_path, signature)
    if (out / "manifest.json").exists():
        done = read_json(out / "manifest.json")
        if done["fingerprint"] != signature["fingerprint"]:
            raise ValueError("已完成的向量库与当前配置不同。")
        verify_bank(out)
        print("复用已完成的向量库：", out)
        return done
    senses = list(read_jsonl(dataset / "senses.jsonl"))
    mapping = {s["sense_id"]: i for i, s in enumerate(senses)}
    sums = np.zeros((len(senses), encoder.dim), dtype=np.float64)
    counts = np.zeros(len(senses), dtype=np.int64)
    total = dm["counts"]["train_examples"]
    with tqdm(total=total, desc="构建义项向量", unit="句") as progress:
        for chunk_id, rows in enumerate(chunks(read_jsonl(dataset / "train.jsonl.gz"), chunk_size)):
            if any(r.get("split") != "train" for r in rows):
                raise ValueError("建库输入中出现非训练样本。")
            row_digest = digest(rows)
            expected_indices = np.array([mapping[r["sense_id"]] for r in rows], dtype=np.int64)
            indices, inverse, expected_counts = np.unique(expected_indices, return_inverse=True, return_counts=True)
            path = ckpt / f"chunk_{chunk_id:06d}.npz"
            cached = False
            if path.exists():
                try:
                    with np.load(path, allow_pickle=False) as z:
                        if str(z["fingerprint"]) != signature["fingerprint"] or str(z["row_digest"]) != row_digest:
                            raise ValueError("检查点输入或模型指纹不匹配。")
                        part_sums, part_counts, part_indices = z["sums"], z["counts"], z["indices"]
                    cached = (np.array_equal(part_indices, indices)
                              and np.array_equal(part_counts, expected_counts)
                              and part_sums.shape == (len(indices), encoder.dim)
                              and np.isfinite(part_sums).all())
                except (OSError, EOFError):
                    cached = False
            if not cached:
                vectors = encoder.encode(rows)
                if vectors.shape != (len(rows), encoder.dim) or not np.isfinite(vectors).all():
                    raise ValueError("编码输出维数或数值错误。")
                part_sums = np.zeros((len(indices), encoder.dim), dtype=np.float64)
                np.add.at(part_sums, inverse, vectors)
                part_counts = expected_counts
                save_npz(path, fingerprint=np.array(signature["fingerprint"]),
                         row_digest=np.array(row_digest), indices=indices,
                         sums=part_sums, counts=part_counts)
            sums[indices] += part_sums
            counts[indices] += part_counts
            progress.update(len(rows))
    expected_counts = np.array([s["n_train"] for s in senses], dtype=np.int64)
    if not np.array_equal(counts, expected_counts) or np.any(counts == 0):
        raise ValueError("每义项的累计训练样本数与数据集不一致。")
    means = (sums / counts[:, None]).astype(np.float32)
    norms = np.linalg.norm(means, axis=1, keepdims=True)
    if not np.isfinite(means).all() or np.any(norms < 1e-12):
        raise ValueError("原型向量为零或包含非有限值。")
    save_npy(out / "prototypes.npy", means)
    save_npy(out / "prototypes_normalized.npy", means / norms)
    for name in ["senses.jsonl", "entries.jsonl"]:
        shutil.copyfile(dataset / name, out / name)
    signature.update({"num_senses": len(senses), "num_train_examples": int(counts.sum()),
                      "sample_mode": dm["sample_mode"], "counts": dm["counts"],
                      "files": {name: file_sha(out / name) for name in
                                ["prototypes.npy", "prototypes_normalized.npy", "senses.jsonl", "entries.jsonl"]}})
    write_json(out / "manifest.json", signature)
    return signature


def verify_bank(path):
    p = Path(path)
    m = read_json(p / "manifest.json")
    if m.get("implementation_sha256") != implementation_digest():
        raise ValueError("当前程序与建库时的程序指纹不同，请使用建库时的代码或重建向量库。")
    for name, expected in m["files"].items():
        if not (p / name).is_file() or file_sha(p / name) != expected:
            raise ValueError(f"向量库文件缺失或已被修改：{name}")
    return m
