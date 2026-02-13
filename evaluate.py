"""
Evaluate spirit-guess language identification on the CommonLID benchmark.

CommonLID uses ISO 639-3 codes; spirit-guess uses ISO 639-1 (with a few 639-3 exceptions).
We build a mapping between the two and evaluate on the intersection.

Usage:
    # Quick balanced eval (100 samples per language):
    python eval_spirit_guess_commonlid.py --sample-per-lang 100 --output results.json

    # Full eval (slow, ~4-5 hours):
    python eval_spirit_guess_commonlid.py --output results.json

    # First N rows only:
    python eval_spirit_guess_commonlid.py --limit 5000

    # Use enchant detector instead of ngram:
    python eval_spirit_guess_commonlid.py --detector enchant --sample-per-lang 100
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict

import pycountry
from datasets import load_dataset


# ---- Language code mapping ----

# Spirit-guess supported codes (from spirit_guess.languages.SUPPORTED_LANGUAGES)
SPIRIT_GUESS_CODES = {
    "af", "ar", "az", "bg", "bn", "bo", "ca", "ceb", "cs", "cy", "da", "de",
    "el", "en", "eo", "es", "et", "eu", "fa", "fi", "fr", "gu", "ha", "haw",
    "he", "hi", "hr", "hu", "hy", "id", "is", "it", "ka", "kk", "km", "ky",
    "la", "lt", "lv", "mk", "ml", "mn", "mr", "nb", "nr", "ne", "nl", "nso",
    "pa", "pl", "ps", "pt", "pt_PT", "pt_BR", "ro", "ru", "sk", "sl", "so",
    "sq", "sr", "ss", "st", "sv", "sw", "te", "th", "tl", "tlh", "tn", "tr",
    "ts", "uk", "ur", "uz", "ve", "vi", "xh", "zu",
}

# Manual mappings for codes pycountry can't resolve automatically.
# Maps CommonLID ISO 639-3 → spirit-guess code.
MANUAL_MAPPING = {
    # Arabic varieties → ar
    "arb": "ar",   # Standard Arabic
    "arz": "ar",   # Egyptian Arabic
    "ary": "ar",   # Moroccan Arabic
    "ars": "ar",   # Najdi Arabic
    "apd": "ar",   # Sudanese Arabic
    "aeb": "ar",   # Tunisian Arabic
    # Individual vs macrolanguage
    "swh": "sw",   # Swahili (individual) → Swahili
    "azj": "az",   # North Azerbaijani → Azerbaijani
    "lvs": "lv",   # Standard Latvian → Latvian
    "uzs": "uz",   # Southern Uzbek → Uzbek
    "zsm": "ms",   # Standard Malay (ms not in spirit-guess, but include for completeness)
    # Close mappings
    "fil": "tl",   # Filipino → Tagalog
    "hbo": "he",   # Ancient Hebrew → Hebrew
    "gaz": "om",   # West Central Oromo → Oromo (om not in spirit-guess)
    # Direct 3-letter code match
    "nso": "nso",  # Pedi — spirit-guess uses this code directly
}


def build_mapping(commonlid_tags):
    """Build CommonLID tag → spirit-guess code mapping."""
    mapping = {}

    for tag in commonlid_tags:
        # Check manual mapping first
        if tag in MANUAL_MAPPING:
            sg_code = MANUAL_MAPPING[tag]
            if sg_code in SPIRIT_GUESS_CODES:
                mapping[tag] = sg_code
            continue

        # Try pycountry automatic lookup
        try:
            lang = pycountry.languages.get(alpha_3=tag)
            if lang and hasattr(lang, "alpha_2"):
                a2 = lang.alpha_2
                if a2 in SPIRIT_GUESS_CODES:
                    mapping[tag] = a2
        except Exception:
            pass

    return mapping


def evaluate(detector_type="ngram", limit=None, sample_per_lang=None,
             output_path=None, seed=42):
    # Load detector
    if detector_type == "ngram":
        from spirit_guess.ngram_detect import NgramDetect
        detector = NgramDetect()
    elif detector_type == "enchant":
        from spirit_guess.enchant_detect import EnchantDetect
        detector = EnchantDetect()
    else:
        print(f"Unknown detector type: {detector_type}")
        sys.exit(1)

    # Load dataset
    print("Loading CommonLID dataset...")
    ds = load_dataset("commoncrawl/CommonLID", split="test", streaming=True)

    all_tags = set()
    rows = []
    print("Streaming dataset...")
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        all_tags.add(row["tag"])
        rows.append(row)
        if (i + 1) % 50000 == 0:
            print(f"  loaded {i + 1} rows...")

    print(f"Total rows loaded: {len(rows)}")
    print(f"Unique tags in data: {len(all_tags)}")

    # Build mapping
    tag_map = build_mapping(all_tags)
    mapped_tags = set(tag_map.keys())
    unmapped_tags = all_tags - mapped_tags
    print(f"Tags mappable to spirit-guess: {len(mapped_tags)}")
    if unmapped_tags:
        print(f"Unmapped tags ({len(unmapped_tags)}): {sorted(unmapped_tags)}")

    # Filter to evaluable rows
    eval_rows = [r for r in rows if r["tag"] in tag_map]
    skipped = len(rows) - len(eval_rows)
    print(f"Evaluable rows: {len(eval_rows)} (skipped {skipped} with unmapped tags)")

    # Optionally subsample per language for balanced, faster evaluation
    if sample_per_lang:
        rng = random.Random(seed)
        by_lang = defaultdict(list)
        for r in eval_rows:
            by_lang[r["tag"]].append(r)
        sampled = []
        for tag, tag_rows in by_lang.items():
            if len(tag_rows) <= sample_per_lang:
                sampled.extend(tag_rows)
            else:
                sampled.extend(rng.sample(tag_rows, sample_per_lang))
        rng.shuffle(sampled)
        print(f"Sampled {len(sampled)} rows ({sample_per_lang}/lang, "
              f"{len(by_lang)} langs)")
        eval_rows = sampled

    # Run evaluation
    print(f"\nRunning {detector_type} detector...")
    correct = 0
    total = 0
    per_lang_correct = Counter()
    per_lang_total = Counter()
    errors = []  # track some wrong predictions for analysis
    unknown_count = 0

    t0 = time.time()
    for i, row in enumerate(eval_rows):
        text = row["text"]
        gold_639_3 = row["tag"]
        gold_639_1 = tag_map[gold_639_3]

        try:
            pred_code, pred_score = detector.detect(text)
        except Exception as e:
            pred_code = "error"
            pred_score = 0.0

        if pred_code == "un":
            unknown_count += 1

        # Normalize: spirit-guess may return pt_BR or pt_PT, map to pt for comparison
        pred_normalized = pred_code.split("_")[0] if "_" in pred_code else pred_code
        gold_normalized = gold_639_1.split("_")[0] if "_" in gold_639_1 else gold_639_1

        is_correct = pred_normalized == gold_normalized
        if is_correct:
            correct += 1
            per_lang_correct[gold_639_3] += 1
        elif len(errors) < 100:
            errors.append({
                "text": text[:200],
                "gold": gold_639_3,
                "gold_639_1": gold_639_1,
                "pred": pred_code,
                "score": pred_score,
            })

        total += 1
        per_lang_total[gold_639_3] += 1

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            acc_so_far = correct / total * 100
            print(f"  {i + 1}/{len(eval_rows)} ({rate:.0f} rows/sec) — accuracy so far: {acc_so_far:.1f}%")

    elapsed = time.time() - t0

    # Compute per-language accuracy
    per_lang_results = {}
    for tag in sorted(per_lang_total.keys(), key=lambda t: -per_lang_total[t]):
        n = per_lang_total[tag]
        c = per_lang_correct[tag]
        per_lang_results[tag] = {
            "total": n,
            "correct": c,
            "accuracy": c / n if n > 0 else 0.0,
            "spirit_guess_code": tag_map[tag],
        }

    # Print results
    print("\n" + "=" * 70)
    print(f"RESULTS — spirit-guess ({detector_type}) on CommonLID")
    print("=" * 70)
    print(f"Overall accuracy: {correct}/{total} = {correct / total * 100:.2f}%")
    print(f"Unknown predictions: {unknown_count}/{total} = {unknown_count / total * 100:.1f}%")
    print(f"Time: {elapsed:.1f}s ({total / elapsed:.0f} rows/sec)")
    print(f"Languages evaluated: {len(per_lang_total)}")
    print()

    # Per-language table
    print(f"{'Tag':<6} {'SG':<5} {'Total':>7} {'Correct':>8} {'Acc':>7}")
    print("-" * 40)
    for tag, info in per_lang_results.items():
        print(f"{tag:<6} {info['spirit_guess_code']:<5} {info['total']:>7} {info['correct']:>8} {info['accuracy']:>6.1%}")

    # Build results dict
    results = {
        "detector": detector_type,
        "overall_accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "unknown_count": unknown_count,
        "skipped_unmapped": skipped,
        "languages_evaluated": len(per_lang_total),
        "elapsed_seconds": elapsed,
        "per_language": per_lang_results,
        "sample_errors": errors[:20],
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nFull results saved to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate spirit-guess on CommonLID")
    parser.add_argument("--detector", choices=["ngram", "enchant"], default="ngram",
                        help="Which spirit-guess detector to use (default: ngram)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max rows to load from dataset (default: all)")
    parser.add_argument("--sample-per-lang", type=int, default=None,
                        help="Max samples per language for balanced eval (default: all)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    evaluate(
        detector_type=args.detector,
        limit=args.limit,
        sample_per_lang=args.sample_per_lang,
        output_path=args.output,
        seed=args.seed,
    )
