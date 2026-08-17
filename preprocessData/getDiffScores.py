# @3vmt/preprocessData/getDiffScores.py
# -*- coding: utf-8 -*-

"""
Compute sentence-level BLEU / COMET / BLEURT for each translation_* in
./data/work3/MMinfoAndTrans/data_with_prompts.json, and save to a new JSON.

Refactored to use utils/computeTransMetric.py for metric calculations.

Run (from repo root):
    python -m 3vmt.preprocessData.getDiffScores \
        --input_json ./data/work3/MMinfoAndTrans/data_with_prompts.json \
        --output_json ./data/work3/MMinfoAndTrans/data_with_prompts_with_metrics.json
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# Import functions from the provided utils
from utils.computeTransMetric import (
    cleanLLMLongTranslate,
    computeBLEU,
    computeCOMET,
    computeBLEURT,
)

TRANSLATION_FIELDS = [
    "translation_baseline",
    "translation_people",
    "translation_objects",
    "translation_actions",
    "translation_ocr",
    "translation_spatial_relations",
    "translation_pointing_gaze",
    "translation_all_cues",
]


@dataclass(frozen=True)
class TripleKey:
    """Dedup key for metric computation."""
    src: str
    ref: str
    pred: str


def _norm_text(s: str) -> str:
    """Normalize text for stable caching."""
    if s is None:
        return ""
    # HTML unescape
    s = html.unescape(str(s))
    # Replace non-breaking spaces
    s = s.replace("\xa0", " ").replace("&nbsp;", " ")
    # Remove residual html tags
    s = re.sub(r"<[^>]+>", "", s)
    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _get_src_ref(item: Dict[str, Any]) -> Tuple[str, str, str, str]:
    """
    Return (src_lang, tgt_lang, src_text, ref_text).
    """
    src_lang = (item.get("src_lang") or "").strip().lower()
    tgt_lang = (item.get("tgt_lang") or "").strip().lower()

    if not src_lang or not tgt_lang:
        lang_field = (item.get("language") or "").strip().lower()
        if lang_field.startswith("en"):
            src_lang = "en"
            tgt_lang = "zh"
        elif lang_field.startswith("zh"):
            src_lang = "zh"
            tgt_lang = "en"
        else:
            raise ValueError(f"Cannot determine source/target language for item with video_id: {item.get('video_id')}")

    if src_lang == "en":
        src_text = item.get("EN_sentence", "")
        ref_text = item.get("ZH_sentence", "")
    elif src_lang == "zh":
        src_text = item.get("ZH_sentence", "")
        ref_text = item.get("EN_sentence", "")
    else:
        raise ValueError(f"Cannot determine source/target language for item with video_id: {item.get('video_id')}")

    return src_lang, tgt_lang, _norm_text(src_text), _norm_text(ref_text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="./data/work3/MMinfoAndTrans/promptsAndTrans/results.json")
    parser.add_argument("--output_json", type=str, default="./data/work3/MMinfoAndTrans/promptsAndTransMetrics.json")
    parser.add_argument("--indent", type=int, default=2)
    # Note: Model paths are hardcoded in utils.computeTransMetric, so they are not exposed here.
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        raise FileNotFoundError(f"Input json not found: {args.input_json}")

    with open(args.input_json, "r", encoding="utf-8") as f:
        data_list: List[Dict[str, Any]] = json.load(f)

    # Step 1: Build per-item mapping -> TripleKey
    triple_to_idx: Dict[TripleKey, int] = {}
    unique_triples: List[TripleKey] = []
    unique_tgt_is_zh: List[bool] = []

    item_field_to_triple: List[Dict[str, TripleKey]] = []

    print("Preprocessing and deduplicating data...")
    for item in data_list:
        _, tgt_lang, src, ref = _get_src_ref(item)
        tgt_is_zh = (tgt_lang == "zh")

        # Clean baseline using util function
        baseline_raw = item.get("translation_baseline", "")
        baseline_pred = _norm_text(cleanLLMLongTranslate(baseline_raw))

        within_cache: Dict[str, TripleKey] = {}
        field_map: Dict[str, TripleKey] = {}

        for field in TRANSLATION_FIELDS:
            raw_pred = item.get(field, "")
            
            # Use util cleaning
            if not raw_pred or raw_pred.strip() == "":
                pred_norm = baseline_pred
            else:
                pred_norm = _norm_text(cleanLLMLongTranslate(raw_pred))

            # Deduplicate
            if pred_norm in within_cache:
                tk = within_cache[pred_norm]
            else:
                tk = TripleKey(src=src, ref=ref, pred=pred_norm)
                within_cache[pred_norm] = tk

            field_map[field] = tk

            if tk not in triple_to_idx:
                triple_to_idx[tk] = len(unique_triples)
                unique_triples.append(tk)
                unique_tgt_is_zh.append(tgt_is_zh)

        item_field_to_triple.append(field_map)

    print(f"Unique triples to evaluate: {len(unique_triples)}")

    # Step 2: Compute metrics for all unique triples
    all_srcs = [t.src for t in unique_triples]
    all_refs = [t.ref for t in unique_triples]
    all_preds = [t.pred for t in unique_triples]

    # --- 2.1 BLEU ---
    # utils.computeBLEU calculates corpus score, so we loop to get sentence scores.
    # Note: computeBLEU expects refs as a list of strings if using SacreBLEU via the util wrapper.
    print("Computing BLEU...")
    bleu_scores: List[float] = []
    for pred, ref, is_zh in zip(all_preds, all_refs, unique_tgt_is_zh):
        # We pass single items as lists to satisfy the list expectation, 
        # but computeBLEU wraps refs in another list internally: refs = [refs].
        # So we just pass the string `ref`.
        score = computeBLEU([pred], [ref], isZh=is_zh)
        bleu_scores.append(float(score))

    # --- 2.2 COMET ---
    # utils.computeCOMET returns a model_output object.
    print("Computing COMET...")
    comet_scores: List[float] = []
    try:
        # computeCOMET handles batching internally (batch_size=8 hardcoded in util)
        comet_output = computeCOMET(all_srcs, all_preds, all_refs)
        # Extract segment scores. Util usually returns 'system_score' * 100 in print, 
        # but the object contains raw 'scores'. We multiply by 100 to match conventions.
        comet_scores = [s * 100.0 for s in comet_output["scores"]]
    except Exception as e:
        print(f"Error computing COMET: {e}")
        comet_scores = [0.0] * len(unique_triples)

    # --- 2.3 BLEURT ---
    # utils.computeBLEURT can return a list if returnAverage=False
    print("Computing BLEURT...")
    try:
        # Returns list of floats. We multiply by 100 to match conventions (0-100 scale).
        bleurt_raw = computeBLEURT(all_preds, all_refs, batchSize=256, returnAverage=False)
        bleurt_scores = [s * 100.0 for s in bleurt_raw]
    except Exception as e:
        print(f"Error computing BLEURT: {e}")
        bleurt_scores = [0.0] * len(unique_triples)

    assert len(bleu_scores) == len(comet_scores) == len(bleurt_scores) == len(unique_triples)

    # Step 3: Pack scores back to each item
    for item, field_map in zip(data_list, item_field_to_triple):
        metrics_obj: Dict[str, Dict[str, float]] = {}
        for field, tk in field_map.items():
            idx = triple_to_idx[tk]
            metrics_obj[field] = {
                "BLEU": bleu_scores[idx],
                "COMET": comet_scores[idx],
                "BLEURT": bleurt_scores[idx],
            }
        item["translation_metrics"] = metrics_obj

    # Step 4: Save
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=args.indent)

    print(f"[Done] Saved: {args.output_json}")
    print(f"[Stats] #items={len(data_list)}, #unique_triples={len(unique_triples)}")


if __name__ == "__main__":
    main()