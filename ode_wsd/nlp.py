from __future__ import annotations

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TreebankWordTokenizer, PunktTokenizer
from .common import normalize


def setup_nltk(download_dir=None):
    for name in ["punkt_tab", "averaged_perceptron_tagger_eng", "wordnet"]:
        ok = nltk.download(name, download_dir=download_dir, quiet=True)
        if not ok:
            raise RuntimeError(f"NLTK 资源下载失败：{name}。请检查 Colab 网络后重试。")
    if download_dir and str(download_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(download_dir))


def wn_pos(tag):
    return {"N": "n", "V": "v", "J": "a", "R": "r"}.get(tag[:1], "n")


def coarse_pos(tag):
    if tag == "MD":
        return "VERB"
    if tag.startswith("NNP"):
        return "PROPN"
    return {"N": "NOUN", "V": "VERB", "J": "ADJ", "R": "ADV"}.get(tag[:1], tag)


class NLTKProcessor:
    def __init__(self):
        self.words = TreebankWordTokenizer()
        self.sentences = PunktTokenizer("english")
        self.lemmatizer = WordNetLemmatizer()
        # Fail early instead of doing a long parse before discovering missing resources.
        self.analyze("The children went home.")

    def sentence_spans(self, text):
        return list(self.sentences.span_tokenize(text))

    def analyze(self, text):
        spans, groups = [], []
        for a, b in self.sentence_spans(text):
            current = [(a + x, a + y) for x, y in self.words.span_tokenize(text[a:b])]
            groups.append([text[x:y] for x, y in current])
            spans.extend(current)
        tagged = [pair for group in nltk.pos_tag_sents(groups, lang="eng") for pair in group]
        if len(tagged) != len(spans):
            raise ValueError("NLTK 字符位置与分词数量不一致。")
        result = []
        for i, ((a, b), (_, tag)) in enumerate(zip(spans, tagged)):
            surface = text[a:b]
            lemma = self.lemmatizer.lemmatize(normalize(surface), wn_pos(tag))
            result.append({"index": i, "text": surface, "start": a, "end": b,
                           "tag": tag, "pos": coarse_pos(tag), "lemma": lemma,
                           "keys": sorted({normalize(surface), normalize(lemma)})})
        return result


def locate_example(text, entry, processor):
    tokens = processor.analyze(text)
    forms = {x["key"] for x in entry["forms"]}
    hits = [t for t in tokens if forms.intersection(t["keys"])]
    if len(hits) != 1:
        return None, "no_target" if not hits else "multiple_targets"
    original = hits[0]
    a, b = next((a, b) for a, b in processor.sentence_spans(text)
                if a <= original["start"] and original["end"] <= b)
    context = text[a:b]
    tokens = processor.analyze(context)
    t = next(x for x in tokens if x["start"] == original["start"] - a
             and x["end"] == original["end"] - a)
    return {"text": context, "source_text": text,
            "source_target_start": original["start"], "source_target_end": original["end"],
            "context_start_in_source": a, "tokens": [x["text"] for x in tokens],
            "target_index": t["index"], "target_start": t["start"], "target_end": t["end"],
            "target_text": t["text"], "target_lemma": t["lemma"],
            "target_pos": t["pos"], "match_keys": t["keys"]}, None
