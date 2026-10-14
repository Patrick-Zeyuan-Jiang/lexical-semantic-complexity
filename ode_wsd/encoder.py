from __future__ import annotations

from dataclasses import asdict
from collections import OrderedDict
import numpy as np
from .common import EncoderConfig


class TargetAlignmentError(ValueError):
    pass


class ContextEncoder:
    """Frozen contextual target vectors: selected layer, mean of target subwords.

    Windowing is centred on the target and includes whole NLTK tokens. No CLS
    pooling, no sentence embeddings, no per-example normalization before mean.
    """
    def __init__(self, config=None, device=None, *, tokenizer=None, model=None):
        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer
        self.config = config or EncoderConfig()
        self._word_count_cache = OrderedDict()
        c = self.config
        if c.batch_size < 1 or c.max_length < 8:
            raise ValueError("batch_size >= 1，max_length >= 8。")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.is_pretrained = model is None
        if model is None:
            hf_config = AutoConfig.from_pretrained(c.model_name, revision=c.revision,
                                                   trust_remote_code=False)
            self.resolved_revision = getattr(hf_config, "_commit_hash", None)
            if not self.resolved_revision:
                raise ValueError("无法取得模型 commit SHA。请使用 Hugging Face 模型名称及明确 revision。")
            self.tokenizer = AutoTokenizer.from_pretrained(
                c.model_name, revision=self.resolved_revision, use_fast=True,
                add_prefix_space=True, trust_remote_code=False)
            self.model = AutoModel.from_pretrained(
                c.model_name, revision=self.resolved_revision, trust_remote_code=False,
                output_hidden_states=True)
        else:
            if tokenizer is None:
                raise ValueError("注入测试模型时须同时提供 tokenizer。")
            self.tokenizer, self.model = tokenizer, model
            self.resolved_revision = "INJECTED_TEST_MODEL"
        if not self.tokenizer.is_fast:
            raise ValueError("必须使用 fast tokenizer，以 word_ids 精确定位子词。")
        if not 1 <= c.layer <= self.model.config.num_hidden_layers:
            raise ValueError("layer 超出 Transformer 层数；0 是 embedding 层，不是第一层。")
        self.limit = min(c.max_length, int(self.model.config.max_position_embeddings),
                         int(self.tokenizer.model_max_length))
        self.budget = self.limit - self.tokenizer.num_special_tokens_to_add(pair=False)
        if self.budget < 1:
            raise ValueError("模型长度不足。")
        self.dim = int(self.model.config.hidden_size)
        self.model.to(self.device).eval()
        self.model.requires_grad_(False)

    @property
    def identity(self):
        import torch, transformers
        return {**asdict(self.config), "revision": self.resolved_revision,
                "effective_max_length": self.limit, "hidden_size": self.dim,
                "subword_pooling": "arithmetic_mean", "example_normalization": "none",
                "prototype_pooling": "arithmetic_mean", "precision": "float32",
                "windowing": "target_centered_whole_words", "eval_mode": True,
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "pretrained": self.is_pretrained}

    def _token_counts(self, words):
        if words in self._word_count_cache:
            self._word_count_cache.move_to_end(words)
            return self._word_count_cache[words]
        tok = self.tokenizer(list(words), is_split_into_words=True, add_special_tokens=False,
                             truncation=False, return_attention_mask=False,
                             verbose=False)
        ids = tok.word_ids()
        counts = np.bincount([i for i in ids if i is not None], minlength=len(words))
        value = counts, len(tok["input_ids"])
        self._word_count_cache[words] = value
        if len(self._word_count_cache) > 512:
            self._word_count_cache.popitem(last=False)
        return value

    def window(self, words, target):
        if not words or not 0 <= target < len(words):
            raise TargetAlignmentError("目标 token 位置越界。")
        counts, length = self._token_counts(tuple(words))
        if counts[target] == 0:
            raise TargetAlignmentError("目标词没有对应的子词。")
        if counts[target] > self.budget:
            raise TargetAlignmentError("目标词自身的子词数超过模型窗口。")
        if length <= self.budget:
            return words, target
        lo, hi = target, target + 1
        total, left, right = int(counts[target]), 0, 0
        while True:
            can_left = lo > 0 and total + counts[lo - 1] <= self.budget
            can_right = hi < len(words) and total + counts[hi] <= self.budget
            if not can_left and not can_right:
                break
            if can_left and (not can_right or left <= right):
                lo -= 1
                total += int(counts[lo])
                left += int(counts[lo])
            else:
                total += int(counts[hi])
                right += int(counts[hi])
                hi += 1
        return words[lo:hi], target - lo

    def encode(self, requests):
        import torch
        requests = list(requests)
        if not requests:
            return np.empty((0, self.dim), dtype=np.float32)
        # Short all-word sentences share one forward pass even with many targets.
        groups = {}
        for i, r in enumerate(requests):
            words, target = self.window(r["tokens"], int(r["target_index"]))
            groups.setdefault(tuple(words), []).append((i, target))
        groups = list(groups.items())
        result = np.empty((len(requests), self.dim), dtype=np.float32)

        def run(batch):
            encoded = self.tokenizer([list(g[0]) for g in batch], is_split_into_words=True,
                                     padding=True, truncation=False, return_tensors="pt")
            if encoded["input_ids"].shape[1] > self.limit:
                raise TargetAlignmentError("窗口重分词后超过长度上限；拒绝截掉目标词。")
            positions = []
            for j, (_, targets) in enumerate(batch):
                word_ids = encoded.word_ids(batch_index=j)
                for original, target in targets:
                    selected = [k for k, wid in enumerate(word_ids) if wid == target]
                    if not selected:
                        raise TargetAlignmentError(f"目标 {original} 在批编码中丢失。")
                    positions.append((j, original, selected))
            try:
                with torch.inference_mode():
                    out = self.model(**{k: v.to(self.device) for k, v in encoded.items()},
                                     output_hidden_states=True, return_dict=True)
                    layer = out.hidden_states[self.config.layer]
                    for j, original, selected in positions:
                        result[original] = layer[j, selected].mean(dim=0).float().cpu().numpy()
                del out, layer
            except torch.cuda.OutOfMemoryError:
                if len(batch) == 1:
                    raise RuntimeError("单句仍然显存不足。请缩小 max_length 并使用新的实验目录。")
                torch.cuda.empty_cache()
                middle = len(batch) // 2
                run(batch[:middle])
                run(batch[middle:])

        for i in range(0, len(groups), self.config.batch_size):
            run(groups[i:i + self.config.batch_size])
        if not np.isfinite(result).all() or np.any(np.linalg.norm(result, axis=1) < 1e-12):
            raise ValueError("编码产生非有限值或零向量，已停止。")
        return result
