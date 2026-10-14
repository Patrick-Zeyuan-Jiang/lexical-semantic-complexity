from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np

from .common import EncoderConfig, normalize, read_jsonl, write_csv
from .bank import verify_bank
from .encoder import ContextEncoder
from .nlp import NLTKProcessor


def same_pos(target, candidate):
    groups = [{"VERB", "AUXILIARY_VERB", "MODAL_VERB"}, {"NOUN", "PROPN"}]
    return target == candidate or any(target in g and candidate in g for g in groups)


class WSDTool:
    def __init__(self, bank_dir, encoder=None, *, processor=None, pos_filter="none"):
        self.path = Path(bank_dir)
        self.manifest = verify_bank(self.path)
        self.senses = list(read_jsonl(self.path / "senses.jsonl"))
        self.entries = {e["entry_key"]: e for e in read_jsonl(self.path / "entries.jsonl")}
        self.sense_map = {s["sense_id"]: s for s in self.senses}
        self.vectors = np.load(self.path / "prototypes_normalized.npy", mmap_mode="r", allow_pickle=False)
        if self.vectors.shape != (len(self.senses), self.manifest["encoder"]["hidden_size"]):
            raise ValueError("向量矩阵与义项索引不一致。")
        self.aliases = defaultdict(set)
        self.alias_kinds = defaultdict(set)
        self.entry_rows = defaultdict(list)
        for entry in self.entries.values():
            for form in entry["forms"]:
                self.aliases[form["key"]].add(entry["entry_key"])
                self.alias_kinds[(form["key"], entry["entry_key"])].add(form["kind"])
        for i, s in enumerate(self.senses):
            self.entry_rows[s["entry_key"]].append(i)
        if pos_filter not in ["none", "strict"]:
            raise ValueError("pos_filter 须为 none 或 strict。")
        self.pos_filter = pos_filter
        # Lazy NLTK/model initialization allows inventory inspection without a GPU.
        self._processor, self._encoder = processor, encoder
        if encoder is not None:
            self._check_encoder(encoder)

    def _check_encoder(self, encoder):
        expected, actual = self.manifest["encoder"], encoder.identity
        for k in ["model_name", "revision", "layer", "effective_max_length", "hidden_size",
                  "subword_pooling", "example_normalization", "precision", "windowing", "pretrained"]:
            if actual.get(k) != expected.get(k):
                raise ValueError(f"WSD 编码器与向量库不匹配：{k}")

    @property
    def encoder(self):
        if self._encoder is None:
            conf = self.manifest["encoder"]
            self._encoder = ContextEncoder(EncoderConfig(**{
                k: conf[k] for k in EncoderConfig.__dataclass_fields__}))
            self._check_encoder(self._encoder)
        return self._encoder

    @property
    def processor(self):
        if self._processor is None:
            self._processor = NLTKProcessor()
        return self._processor

    def candidates(self, request, *, restrict_entry=None):
        if restrict_entry is not None:
            entries = {restrict_entry} if restrict_entry in self.entries else set()
            kinds = ["gold_entry_oracle"]
        else:
            keys = request["match_keys"]
            entries = set().union(*(self.aliases.get(k, set()) for k in keys))
            kinds = sorted({kind for k in keys for e in entries
                            for kind in self.alias_kinds.get((k, e), set())})
        indices = sorted({i for entry in entries for i in self.entry_rows[entry]})
        if self.pos_filter == "strict":
            indices = [i for i in indices if same_pos(request.get("target_pos", ""), self.senses[i]["pos"])]
        return indices, sorted(entries), kinds

    def rank(self, request, vector, *, top_k=3, restrict_entry=None):
        indices, entries, kinds = self.candidates(request, restrict_entry=restrict_entry)
        info = {"status": "ok" if indices else "no_candidate", "matched_entries": entries,
                "match_kinds": kinds, "candidate_count": len(indices),
                "candidate_main_count": 0, "main_sense_id": None, "fine_sense_id": None,
                "cosine": None, "main_margin": None, "top_candidates": []}
        if not indices:
            return info
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != (self.vectors.shape[1],) or not np.isfinite(vector).all() or norm < 1e-12:
            raise ValueError("目标向量维数/数值错误。")
        scores = self.vectors[indices] @ (vector / norm)
        # Stable tie breaker makes results reproducible; no hidden frequency prior.
        order = sorted(range(len(indices)), key=lambda j: (-float(scores[j]), self.senses[indices[j]]["sense_id"]))
        main_best = {}
        fine_rank = []
        for j in order:
            s = self.senses[indices[j]]
            item = {"fine_sense_id": s["sense_id"], "main_sense_id": s["main_sense_id"],
                    "cosine": float(scores[j]), "definition": s["definition"]}
            main_best.setdefault(s["main_sense_id"], item)
            fine_rank.append(item)
        winner = fine_rank[0]
        main_order = list(main_best.values())
        parent = self.sense_map[winner["main_sense_id"]]
        info.update({"main_sense_id": winner["main_sense_id"], "fine_sense_id": winner["fine_sense_id"],
                     "cosine": winner["cosine"], "definition": parent["definition"],
                     "fine_definition": winner["definition"], "lemma": parent["lemma"],
                     "candidate_main_count": len(main_best),
                     "main_margin": main_order[0]["cosine"] - main_order[1]["cosine"] if len(main_order) > 1 else None,
                     "top_candidates": main_order[:max(0, top_k)]})
        return info

    def predict_requests(self, requests, *, top_k=3):
        requests = list(requests)
        results = [None] * len(requests)
        covered = [i for i, r in enumerate(requests) if self.candidates(r)[0]]
        covered_set = set(covered)
        for i, r in enumerate(requests):
            if i not in covered_set:
                results[i] = self.rank(r, np.zeros(self.vectors.shape[1]), top_k=top_k)
        if covered:
            vectors = self.encoder.encode([requests[i] for i in covered])
            for i, vector in zip(covered, vectors):
                results[i] = self.rank(requests[i], vector, top_k=top_k)
        return results

    @staticmethod
    def _request(text, tokens, token):
        return {"text": text, "tokens": [t["text"] for t in tokens],
                "target_index": token["index"], "target_start": token["start"],
                "target_end": token["end"], "target_text": token["text"],
                "target_lemma": token["lemma"], "target_pos": token["pos"],
                "match_keys": token["keys"]}

    def disambiguate(self, text, *, include_uncovered=False, top_k=3):
        requests, display = [], []
        for sentence_index, (a, b) in enumerate(self.processor.sentence_spans(text)):
            sentence = text[a:b]
            tokens = self.processor.analyze(sentence)
            for token in tokens:
                if not any(c.isalpha() for c in token["text"]):
                    continue
                r = self._request(sentence, tokens, token)
                if not include_uncovered and not self.candidates(r)[0]:
                    continue
                requests.append(r)
                display.append({"sentence_index": sentence_index, "sentence": sentence,
                                "target": token["text"], "lemma_from_nltk": token["lemma"],
                                "pos_from_nltk": token["pos"],
                                "start": a + token["start"], "end": a + token["end"]})
        return [{**view, **prediction} for view, prediction in
                zip(display, self.predict_requests(requests, top_k=top_k))]

    def predict_target(self, text, target=None, *, start=None, end=None, occurrence=None, top_k=3):
        tokens = self.processor.analyze(text)
        if start is not None or end is not None:
            if start is None or end is None:
                raise ValueError("start/end 必须同时提供；字符区间为 [start,end)。")
            hits = [t for t in tokens if t["start"] == start and t["end"] == end]
            if target is not None and text[start:end] != target:
                raise ValueError("target 与指定字符区间不一致。")
        else:
            if target is None:
                raise ValueError("请提供 target 或精确字符位置。")
            hits = [t for t in tokens if normalize(t["text"]) == normalize(target)]
        if occurrence is not None:
            if not 0 <= occurrence < len(hits):
                raise ValueError("occurrence 超出出现次数；从 0 开始计数。")
            hits = [hits[occurrence]]
        if len(hits) != 1:
            raise ValueError(f"找到 {len(hits)} 个目标；请指定唯一字符位置或 occurrence。")
        t = hits[0]
        # User text may contain several sentences: restrict to target's sentence,
        # exactly as in disambiguate(), while maintaining original character offsets.
        a, b = next((a, b) for a, b in self.processor.sentence_spans(text)
                    if a <= t["start"] and t["end"] <= b)
        local = self.processor.analyze(text[a:b])
        selected = next(x for x in local if x["start"] == t["start"] - a and x["end"] == t["end"] - a)
        r = self._request(text[a:b], local, selected)
        return {"target": t["text"], "start": t["start"], "end": t["end"],
                **self.predict_requests([r], top_k=top_k)[0]}

    def tag_file(self, input_path, output_path, *, include_uncovered=True):
        p = Path(input_path)
        rows = self.disambiguate(p.read_text(encoding="utf-8-sig"), include_uncovered=include_uncovered)
        write_csv(output_path, [dict(file=p.name, **r) for r in rows], fields=[
            "file", "sentence_index", "sentence", "target", "lemma_from_nltk", "pos_from_nltk",
            "start", "end", "status", "matched_entries", "match_kinds", "candidate_count",
            "candidate_main_count", "main_sense_id", "fine_sense_id", "cosine", "main_margin",
            "top_candidates", "definition", "fine_definition", "lemma"])
        return rows
