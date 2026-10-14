from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import DataConfig, EncoderConfig, find_export


def main():
    parser = argparse.ArgumentParser(description="ODE GB 向量库、WSD 和评测")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("nltk", help="下载 NLTK 英文资源")
    p = sub.add_parser("prepare", help="解析和划分完整 ODE 导出")
    p.add_argument("--input")
    p.add_argument("--source-dir", default="/content/drive/MyDrive/9.10词汇语义复杂度/ODE glance")
    p.add_argument("--output", required=True)
    p.add_argument("--sample-mode", action="store_true")
    p.add_argument("--seed", type=int, default=20260929)
    p.add_argument("--min-usable", type=int, default=2)
    p.add_argument("--no-derivatives", action="store_true")
    p = sub.add_parser("build", help="构建/续建义项原型")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model", default="microsoft/deberta-v3-base")
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--max-length", type=int, default=512)
    p = sub.add_parser("evaluate", help="评测留出集")
    p.add_argument("--dataset", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pos-filter", choices=["none", "strict"], default="none")
    p = sub.add_parser("tag", help="处理 UTF-8 文本文件")
    p.add_argument("--bank", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p = sub.add_parser("evaluate-external", help="评测已人工标注的 CSV")
    p.add_argument("--bank", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "nltk":
        from .nlp import setup_nltk
        setup_nltk()
    elif args.command == "prepare":
        from .dataset import prepare_dataset
        prepare_dataset(args.input or find_export(args.source_dir), args.output,
                        DataConfig(seed=args.seed, min_usable_examples=args.min_usable,
                                   include_derivatives=not args.no_derivatives), sample_mode=args.sample_mode)
    elif args.command == "build":
        from .bank import build_bank
        build_bank(args.dataset, args.output, config=EncoderConfig(
            model_name=args.model, layer=args.layer, batch_size=args.batch_size, max_length=args.max_length))
    elif args.command == "evaluate":
        from .wsd import WSDTool
        from .evaluate import evaluate_holdout
        evaluate_holdout(args.dataset, WSDTool(args.bank, pos_filter=args.pos_filter), args.output)
    elif args.command == "tag":
        from .wsd import WSDTool
        rows = WSDTool(args.bank).tag_file(args.input, args.output)
        print(f"已导出 {len(rows)} 条 token 结果：{args.output}")
    elif args.command == "evaluate-external":
        from .wsd import WSDTool
        from .evaluate import evaluate_annotated_csv
        result = evaluate_annotated_csv(WSDTool(args.bank), args.input, args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

