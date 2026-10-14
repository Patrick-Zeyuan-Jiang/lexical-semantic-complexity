"""Parser for the observed ode_glance HTML, never executes dictionary JavaScript."""
from __future__ import annotations

import re
from collections import Counter
from bs4 import BeautifulSoup

from .common import DataConfig, deaccent, normalize, one_word, sentence_key


def classes(tag):
    return set(tag.get("class", [])) if tag else set()


def text(tag):
    return " ".join(tag.stripped_strings).strip() if tag else ""


def nearest_class(node, name):
    return node.find_parent(lambda t: name in classes(t))


def is_dictionary_example(node, sense):
    if nearest_class(node, "msDict") is not sense:
        return False
    return not node.find_parent(lambda t: bool(classes(t) & {
        "entrySynList", "kthsr", "thesaurus", "cn", "cn_def", "cn_def_text"}))


POS = {"noun": "NOUN", "proper noun": "PROPN", "verb": "VERB",
       "adjective": "ADJ", "adverb": "ADV", "pronoun": "PRON",
       "preposition": "ADP", "conjunction": "CONJ", "determiner": "DET",
       "exclamation": "INTJ", "interjection": "INTJ", "number": "NUM",
       "numeral": "NUM", "abbreviation": "ABBREVIATION", "symbol": "SYMBOL",
       "modal verb": "MODAL_VERB", "auxiliary verb": "AUXILIARY_VERB"}


def canonical_pos(value):
    value = " ".join(value.casefold().split())
    return POS.get(value, re.sub(r"\W+", "_", value.upper()).strip("_"))


def expand_forms(forms):
    """Keep both visible surface forms and provenance, not an alias->single-entry map."""
    result = {}
    for f in forms:
        value = f["form"].strip()
        if not one_word(value) or value.casefold() == "same":
            continue
        for variant, suffix in [(value, ""), (value.lower(), ":lower"),
                                (deaccent(value), ":deaccent"),
                                (deaccent(value).lower(), ":deaccent:lower")]:
            key = (variant, f["kind"] + suffix, f.get("source_pos", ""))
            result[key] = {"form": variant, "key": normalize(variant),
                           "kind": key[1], "source_pos": key[2]}
    return sorted(result.values(), key=lambda x: (x["form"], x["kind"], x["source_pos"]))


def parse_record(record, config=None):
    """Return (entries, senses, diagnostics); one MDX record may contain homographs.

    Sense numbers come from the first own .iteration. Single unnumbered main
    senses receive 1 with number_inferred=True; never renumber retained senses.
    """
    config = config or DataConfig()
    counts = Counter()
    if record.get("decode_error"):
        raise ValueError(f"解码错误记录 {record.get('entry_id')}")
    html = record.get("html", "")
    if record.get("redirect_target") or html.lstrip().startswith("@@@LINK="):
        return [], [], {"redirect_record": 1}
    soup = BeautifulSoup(html, "lxml")
    halves = soup.select("halffolder.karxUK")
    if not halves:
        return [], [], {"no_GB_section": 1}
    entries, senses = {}, []
    for half in halves:
        for page in half.select(".entryPageContent"):
            title = page.select_one("header.entryHeader .pageTitle .realTitle")
            lemma = text(title)
            if not one_word(lemma):
                counts["non_single_word_page"] += 1
                continue
            sections = [s for s in page.select("section.se1.senseGroup")
                        if nearest_class(s, "entryPageContent") is page
                        and not nearest_class(s, "subEntryBlock")]
            if not sections:
                counts["GB_page_without_main_senses"] += 1
                continue
            hom = text(page.select_one("header.entryHeader .pageTitle .homograph"))
            inferred_hom = not hom
            if not hom:
                hom = "1"
            if not re.fullmatch(r"\d+", hom):
                raise ValueError(f"无法识别同形词类别号：{lemma} {hom!r}")
            forms = [{"form": lemma, "kind": "headword"}]
            # These groups are siblings after the header, not necessarily within it.
            for group in page.select(".variantGroup"):
                if any(nearest_class(group, k) for k in
                       ["senseGroup", "subEntryBlock", "msDict", "entrySynList"]):
                    continue
                kind = "US_spelling" if "US English" in text(group) else "variant"
                forms += [{"form": text(v), "kind": kind}
                          for v in group.select(".variant")]
            if config.include_derivatives:
                for derivative in page.select(".derivatives .subEntry"):
                    p = canonical_pos(text(derivative.select_one("dd > .partOfSpeech")))
                    forms += [{"form": text(v), "kind": "derivative", "source_pos": p}
                              for v in derivative.select("dt .h4SubSense .l")]
                    for group in derivative.select(".variantGroup"):
                        kind = "derivative_US" if "US English" in text(group) else "derivative_variant"
                        forms += [{"form": text(v), "kind": kind, "source_pos": p}
                                  for v in group.select(".variant")]
            for sec in sections:
                pos_original = text(sec.select_one("h2.partOfSpeechTitle > .partOfSpeech"))
                pos = canonical_pos(pos_original) or "UNSPECIFIED"
                if not pos_original:
                    counts["sections_without_explicit_POS"] += 1
                if config.include_inflections:
                    forms += [{"form": text(v), "kind": "inflection", "source_pos": pos}
                              for v in sec.select("h2.partOfSpeechTitle .inflection")
                              if "same" not in classes(v)]
                nodes = [s for s in sec.select(".msDict")
                         if nearest_class(s, "senseGroup") is sec
                         and not nearest_class(s, "entrySynList")]
                main_count = sum("sense" in classes(s) for s in nodes)
                latest_main = None
                for node in nodes:
                    is_sub = "subsense" in classes(node)
                    if not is_sub and "sense" not in classes(node):
                        continue
                    own = lambda selector: [n for n in node.select(selector)
                                             if nearest_class(n, "msDict") is node
                                             and not nearest_class(n, "entrySynList")]
                    numbers = own(".iteration")
                    number = text(numbers[0]) if numbers else ""
                    inferred_number = not number
                    if not number and not is_sub and main_count == 1:
                        number = "1"
                    if not re.fullmatch(r"\d+(?:\.\d+)*", number):
                        raise ValueError(f"未能可靠提取义项号：{lemma}/{hom}/{pos}/{number!r}")
                    if is_sub and "." not in number:
                        raise ValueError(f"次义项没有分级编号：{lemma}/{number}")
                    if not is_sub and "." in number:
                        raise ValueError(f"主义项编号异常：{lemma}/{number}")
                    parent_num = number.split(".")[0] if is_sub else number
                    if is_sub and latest_main != parent_num:
                        raise ValueError(f"次义项父节点与编号不符：{lemma}/{pos}/{number}")
                    if not is_sub:
                        latest_main = number
                    defs = own(".definition")
                    definition = text(defs[0]) if defs else ""
                    if not definition:
                        counts["sense_without_definition"] += 1
                        continue
                    examples = {}
                    for ex in node.select("em.example, li.sentence"):
                        if not is_dictionary_example(ex, node):
                            continue
                        t = text(ex)
                        if t:
                            examples.setdefault(sentence_key(t), {
                                "text": t, "source_type": "inline" if ex.name == "em" else "more_examples"})
                    anchors = own("a[myid]")
                    source = {"record_id": record.get("entry_id"),
                              "mdx_headword": record.get("headword"),
                              "anchor": anchors[0].get("myid") if anchors else None}
                    senses.append({"sense_id": f"{lemma}_{hom}_{pos}_{number}",
                                   "entry_key": lemma, "lemma": lemma, "homograph": hom,
                                   "pos": pos, "pos_original": pos_original,
                                   "number": number, "level": 2 if is_sub else 1,
                                   "main_sense_id": f"{lemma}_{hom}_{pos}_{parent_num}",
                                   "definition": definition, "number_inferred": inferred_number,
                                   "homograph_inferred": inferred_hom,
                                   "examples": list(examples.values()), "sources": [source]})
            entry = entries.setdefault(lemma, {"entry_key": lemma, "lemma": lemma, "forms": []})
            entry["forms"].extend(forms)
    for entry in entries.values():
        entry["forms"] = expand_forms(entry["forms"])
    counts["GB_pages_parsed"] += len(entries)
    counts["parsed_senses"] += len(senses)
    return list(entries.values()), senses, dict(counts)
