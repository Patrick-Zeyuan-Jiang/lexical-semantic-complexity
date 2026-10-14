from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path

VERSION = "1.0.0"


@dataclass(frozen=True)
class DataConfig:
    min_raw_examples: int = 5
    # 原设计在定位目标词之后才丢弃例句；至少保留一条训练、一条测试。
    min_usable_examples: int = 2
    min_main_senses: int = 2
    test_fraction: float = 0.10
    seed: int = 20260929
    include_derivatives: bool = True
    include_inflections: bool = True

    def validate(self):
        if self.min_raw_examples < 2 or self.min_usable_examples < 2:
            raise ValueError("例句下限必须至少为 2。")
        if self.min_main_senses < 2 or not 0 < self.test_fraction < 1:
            raise ValueError("至少两个主义项；测试比例须在 0 和 1 之间。")


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "microsoft/deberta-v3-base"
    revision: str = "main"  # 初始化时解析成不可变的模型 commit SHA
    layer: int = 7  # hidden_states[0] 是 embedding，7 是第七个 Transformer 层
    max_length: int = 512
    batch_size: int = 16


def normalize(text):
    """Only matching keys are normalized; encoder inputs retain original case/accent."""
    return unicodedata.normalize("NFC", text).replace("’", "'").casefold().strip()


def deaccent(text):
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def one_word(text):
    # 包括常见 Unicode 连字符；至少含一个字母，排除纯符号条目。
    return bool(text and any(c.isalpha() for c in text)
                and not re.search(r"[\s\-‐‑‒–—―]", text))


def sentence_key(text):
    return normalize(" ".join(text.split()))


def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def implementation_digest():
    return digest({p.name: file_sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))})


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def read_jsonl(path):
    path = Path(path)
    op = gzip.open if path.suffix == ".gz" else open
    with op(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except Exception as e:
                    raise ValueError(f"JSONL 第 {i} 行解析失败：{path}") from e


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    op = gzip.open if path.suffix == ".gz" else open
    with op(tmp, "wt", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for r in rows for k in r))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list))
                        else v for k, v in r.items()})


def find_export(source_dir):
    """Select the newest complete export made by the user's unpack notebook."""
    parent = Path(source_dir).parent / "ODE_拆包结果"
    candidates = []
    for p in parent.rglob("ode_entries.jsonl.gz"):
        report = p.parent / "report.json"
        if not report.exists():
            continue
        r = read_json(report)
        if r.get("mdx", {}).get("complete") and r["mdx"].get("exported_entries", 0):
            candidates.append((r.get("run_id_utc", ""), p))
    if not candidates:
        raise FileNotFoundError(
            f"未找到完整导出。请先运行原拆包 Notebook，或手动指定完整 ode_entries.jsonl.gz；"
            f"检查目录：{parent}。不要用诊断包 samples.jsonl 代替全量数据。")
    return sorted(candidates, key=lambda x: x[0])[-1][1]


def validate_export(path, sample_mode=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if sample_mode:
        return None
    if path.name != "ode_entries.jsonl.gz":
        raise ValueError("正式模式只接受完整 ode_entries.jsonl.gz。样本测试须显式 sample_mode=True。")
    report = path.parent / "report.json"
    if not report.exists():
        raise ValueError("完整导出旁须有拆包生成的 report.json，用于核验完整性。")
    r = read_json(report)
    m = r.get("mdx", {})
    if not m.get("complete") or m.get("exported_entries") != m.get("expected_entries"):
        raise ValueError("拆包报告没有确认完整导出，请先修复拆包。")
    if m.get("decode_error_entries", 0):
        raise ValueError("拆包存在解码错误，须先检查原始记录编码。")
    return int(m["exported_entries"])


def config_dict(config):
    return asdict(config)
