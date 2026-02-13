# spirit-guess evaluation on CommonLID

Evaluation of the [spirit-guess](https://github.com/AidanSun05/spirit-guess) language identification library against the [CommonLID](https://huggingface.co/datasets/commoncrawl/CommonLID) benchmark dataset.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Balanced eval (200 samples per language, ~20 min):
python evaluate.py --sample-per-lang 200 --output results/ngram_200_per_lang.json

# Full eval (slow, ~4-5 hours):
python evaluate.py --output results/full.json

# Use enchant detector instead of ngram:
python evaluate.py --detector enchant --sample-per-lang 200
```

CommonLID uses ISO 639-3 codes; spirit-guess uses ISO 639-1. The script builds a mapping between the two and evaluates on the intersection (55 of ~200 CommonLID languages).

## Results

**ngram detector, 200 samples per language, 55 languages evaluated**

| Metric | Value |
|---|---|
| Overall accuracy | 62.3% (5,476 / 8,791) |
| Unknown predictions | 6.6% (583 / 8,791) |
| Languages evaluated | 55 |
| Languages with >80% accuracy | 24 |
| Languages with 0% accuracy | 11 |

### Per-language breakdown

#### Top performers (>85% accuracy)

| Language | Code | Samples | Accuracy |
|---|---|---|---|
| Standard Arabic | arb | 200 | 95.5% |
| North Azerbaijani | azj | 200 | 95.0% |
| Southern Sotho | sot | 200 | 94.5% |
| Najdi Arabic | ars | 200 | 93.0% |
| Persian | fas | 200 | 92.5% |
| Arabic | ara | 200 | 91.0% |
| Sudanese Arabic | apd | 27 | 88.9% |
| Vietnamese | vie | 200 | 89.5% |
| Swahili (individual) | swh | 200 | 88.5% |
| Egyptian Arabic | arz | 200 | 88.0% |
| Finnish | fin | 200 | 87.5% |
| Latvian (standard) | lvs | 200 | 87.5% |
| Catalan | cat | 93 | 87.1% |
| Tunisian Arabic | aeb | 15 | 86.7% |
| Indonesian | ind | 200 | 87.0% |
| Italian | ita | 200 | 86.5% |
| Azerbaijani | aze | 29 | 86.2% |
| French | fra | 200 | 86.0% |
| Hausa | hau | 200 | 85.5% |
| Latvian | lav | 200 | 85.5% |

#### Mid-range (40-85%)

| Language | Code | Samples | Accuracy |
|---|---|---|---|
| German | deu | 200 | 84.0% |
| Spanish | spa | 200 | 84.0% |
| Turkish | tur | 200 | 83.5% |
| Dutch | nld | 200 | 83.0% |
| Hindi | hin | 200 | 83.0% |
| Urdu | urd | 200 | 80.0% |
| Bulgarian | bul | 109 | 78.0% |
| Swahili (macrolanguage) | swa | 200 | 78.5% |
| Moroccan Arabic | ary | 200 | 76.0% |
| English | eng | 200 | 75.5% |
| Russian | rus | 200 | 74.5% |
| Estonian | est | 200 | 74.5% |
| Latin | lat | 50 | 74.0% |
| Portuguese | por | 200 | 73.5% |
| Xhosa | xho | 200 | 68.5% |
| Northern Sotho | nso | 99 | 67.7% |
| Afrikaans | afr | 88 | 65.9% |
| Ukrainian | ukr | 2 | 50.0% |
| Filipino | fil | 58 | 43.1% |
| Tagalog | tgl | 200 | 39.5% |
| Czech | ces | 200 | 37.5% |

#### Complete failures (0% accuracy)

| Language | Code | Samples | Script |
|---|---|---|---|
| Malayalam | mal | 200 | Malayalam |
| Gujarati | guj | 200 | Gujarati |
| Hebrew | heb | 200 | Hebrew |
| Ancient Hebrew | hbo | 200 | Hebrew |
| Punjabi | pan | 200 | Gurmukhi |
| Telugu | tel | 200 | Telugu |
| Bengali | ben | 200 | Bengali |
| Marathi | mar | 200 | Devanagari |
| Thai | tha | 200 | Thai |
| Greek | ell | 7 | Greek |
| Southern Uzbek | uzs | 9 | Arabic |

### Key findings

1. **Non-Latin scripts are completely unsupported.** Languages written in Malayalam, Gujarati, Hebrew, Telugu, Bengali, Devanagari, Thai, and Greek scripts all score 0%. The ngram detector appears to only handle Latin, Cyrillic, and Arabic scripts. Notably, Hindi (Devanagari) scores 83% while Marathi (also Devanagari) scores 0%, suggesting Hindi may be detected by other heuristics.

2. **Arabic dialect handling works well.** All Arabic varieties (Standard, Egyptian, Moroccan, Najdi, Sudanese, Tunisian) are correctly mapped to `ar` and achieve 76-96% accuracy.

3. **Closely related languages cause confusion.** Czech is frequently misidentified as Slovak (37.5% accuracy). Tagalog/Filipino are confused with Cebuano. Dutch is sometimes classified as Afrikaans.

4. **Uzbek (Latin script) nearly completely fails** (0.5% accuracy), with predictions scattered across Hausa, Croatian, and other languages. This suggests spirit-guess lacks adequate Uzbek training data.

5. **6.6% of predictions are "unknown"**, indicating the detector returns no confident match for a meaningful fraction of inputs.

## License

MIT
