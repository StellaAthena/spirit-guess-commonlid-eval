# spirit-guess evaluation on CommonLID

Evaluation of the [spirit-guess](https://github.com/alvations/spirit-guess) language identification library against the [CommonLID](https://huggingface.co/datasets/commoncrawl/CommonLID) benchmark dataset.

CommonLID ([Ortiz Suarez et al., 2025](https://arxiv.org/abs/2601.18026)) is a 373,230-line benchmark of web-crawled text spanning 109 language varieties. It was designed to stress-test language identification on noisy, real-world web data --- a much harder setting than clean benchmarks like FLORES+.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Balanced eval (200 samples per language, ~20 min):
python evaluate.py --sample-per-lang 200 --output results/ngram_200_per_lang.json

# Full eval (slow, ~4-5 hours):
python evaluate.py --output results/ngram_full.json

# Use enchant detector instead of ngram:
python evaluate.py --detector enchant --sample-per-lang 200
```

CommonLID uses ISO 639-3 codes; spirit-guess uses ISO 639-1. The script builds a mapping between the two and evaluates on the intersection (55 of 109 CommonLID languages).

## Results

### Summary

**ngram detector, full test set (373,230 lines, 109 languages)**

| Metric | Value |
|---|---|
| Micro accuracy (mapped languages only) | 66.0% (195,700 / 296,363) |
| Micro accuracy (full dataset) | 52.4% (195,700 / 373,230) |
| Macro-averaged recall (mapped only, 55 langs) | 61.5% |
| Macro-averaged recall (all 109 langs) | 31.0% |
| Unknown predictions | 5.6% (16,734 / 296,363) |
| Languages covered | 55 / 109 |
| Languages with >80% accuracy | 25 |
| Languages with 0% accuracy | 11 |

### Comparison to other models

The CommonLID paper evaluates 8 dedicated LID tools and 4 LLMs. The table below compares spirit-guess against them. The paper reports **macro-averaged F1**; our evaluation measures **macro-averaged recall** (per-language accuracy). Since F1 incorporates precision, the metrics are not directly comparable, but the comparison is directionally informative. Spirit-guess's F1 would require the full confusion matrix to compute exactly.

For both the paper's models and spirit-guess, the "all" column scores unsupported languages as 0.

| Model | Languages covered | Macro F1 (all) | Macro F1 (covered) | Source |
|---|---|---|---|---|
| GlotLID v4 | 96 / 109 | **60.4** | 68.6 | CommonLID paper |
| CLD2 | 68 / 109 | 49.5 | **79.3** | CommonLID paper |
| fastText (NLLB) | 74 / 109 | 49.3 | 72.6 | CommonLID paper |
| OpenLID-v2 | 76 / 109 | 47.4 | 68.0 | CommonLID paper |
| FUN-LangID | 86 / 109 | 46.4 | 57.8 | CommonLID paper |
| pyFranc | 70 / 109 | 39.3 | 61.3 | CommonLID paper |
| CLD3 | 55 / 109 | 34.3 | 66.2 | CommonLID paper |
| **spirit-guess (ngram)** | **55 / 109** | **31.0\*** | **61.5\*** | This eval |
| AfroLID | 23 / 109 | 9.2 | 43.5 | CommonLID paper |

\* Macro-averaged recall, not F1. Actual F1 would be lower if precision < recall for most languages.

Spirit-guess has the same language coverage as CLD3 (55 languages) and achieves comparable performance on covered languages (61.5% recall vs 66.2% F1). Both fall well short of higher-coverage models like GlotLID v4. The primary bottleneck for spirit-guess is coverage: with only 55 of 109 languages supported, 54 languages contribute 0 to the macro average.

The CommonLID paper also evaluates LLMs on a pairwise basis with GlotLID. On the 76-language "Core" set shared by all dedicated tools, GPT-5 achieves 91.8 F1 and GPT-4o achieves 89.0 F1, compared to GlotLID's 93.5 F1.

### Per-language breakdown

#### Top performers (>80% accuracy)

| Language | ISO 639-3 | spirit-guess code | Samples | Correct | Accuracy |
|---|---|---|---|---|---|
| Southern Sotho | sot | st | 943 | 891 | 94.5% |
| Persian | fas | fa | 19,318 | 18,246 | 94.5% |
| Standard Arabic | arb | ar | 26,152 | 24,640 | 94.2% |
| Najdi Arabic | ars | ar | 229 | 212 | 92.6% |
| Vietnamese | vie | vi | 21,803 | 19,918 | 91.4% |
| Arabic | ara | ar | 16,306 | 14,818 | 90.9% |
| North Azerbaijani | azj | az | 847 | 754 | 89.0% |
| Indonesian | ind | id | 33,828 | 30,137 | 89.1% |
| Swahili (individual) | swh | sw | 12,383 | 11,095 | 89.6% |
| Sudanese Arabic | apd | ar | 27 | 24 | 88.9% |
| Egyptian Arabic | arz | ar | 1,102 | 976 | 88.6% |
| Hausa | hau | ha | 16,455 | 14,390 | 87.5% |
| Latvian | lav | lv | 512 | 448 | 87.5% |
| Catalan | cat | ca | 93 | 81 | 87.1% |
| Tunisian Arabic | aeb | ar | 15 | 13 | 86.7% |
| Hindi | hin | hi | 3,666 | 3,165 | 86.3% |
| Azerbaijani | aze | az | 29 | 25 | 86.2% |
| Finnish | fin | fi | 1,030 | 887 | 86.1% |
| French | fra | fr | 3,233 | 2,775 | 85.8% |
| Turkish | tur | tr | 4,486 | 3,845 | 85.7% |
| Italian | ita | it | 4,387 | 3,735 | 85.1% |
| German | deu | de | 7,553 | 6,423 | 85.0% |
| Standard Latvian | lvs | lv | 353 | 300 | 85.0% |
| Dutch | nld | nl | 3,299 | 2,705 | 82.0% |
| Urdu | urd | ur | 204 | 164 | 80.4% |

#### Mid-range (30-80%)

| Language | ISO 639-3 | spirit-guess code | Samples | Correct | Accuracy |
|---|---|---|---|---|---|
| Spanish | spa | es | 4,236 | 3,384 | 79.9% |
| Swahili (macrolanguage) | swa | sw | 4,031 | 3,213 | 79.7% |
| Portuguese | por | pt | 2,443 | 1,932 | 79.1% |
| Bulgarian | bul | bg | 109 | 85 | 78.0% |
| Moroccan Arabic | ary | ar | 226 | 174 | 77.0% |
| English | eng | en | 27,461 | 20,874 | 76.0% |
| Latin | lat | la | 50 | 37 | 74.0% |
| Russian | rus | ru | 4,003 | 2,966 | 74.1% |
| Estonian | est | et | 659 | 484 | 73.4% |
| Northern Sotho | nso | nso | 99 | 67 | 67.7% |
| Afrikaans | afr | af | 88 | 58 | 65.9% |
| Xhosa | xho | xh | 575 | 363 | 63.1% |
| Ukrainian | ukr | uk | 2 | 1 | 50.0% |
| Filipino | fil | tl | 58 | 25 | 43.1% |
| Tagalog | tgl | tl | 2,223 | 867 | 39.0% |
| Czech | ces | cs | 933 | 353 | 37.8% |
| Zulu | zul | zu | 4 | 1 | 25.0% |

#### Near-zero and zero accuracy

| Language | ISO 639-3 | spirit-guess code | Samples | Correct | Accuracy | Script |
|---|---|---|---|---|---|---|
| Uzbek | uzb | uz | 43,189 | 148 | 0.3% | Latin |
| Telugu | tel | te | 11,747 | 0 | 0.0% | Telugu |
| Hebrew | heb | he | 5,055 | 0 | 0.0% | Hebrew |
| Thai | tha | th | 3,118 | 0 | 0.0% | Thai |
| Malayalam | mal | ml | 2,061 | 0 | 0.0% | Malayalam |
| Bengali | ben | bn | 1,886 | 0 | 0.0% | Bengali |
| Marathi | mar | mr | 1,061 | 0 | 0.0% | Devanagari |
| Punjabi | pan | pa | 1,020 | 0 | 0.0% | Gurmukhi |
| Gujarati | guj | gu | 948 | 0 | 0.0% | Gujarati |
| Ancient Hebrew | hbo | he | 808 | 0 | 0.0% | Hebrew |
| Southern Uzbek | uzs | uz | 9 | 0 | 0.0% | Arabic |
| Greek | ell | el | 7 | 0 | 0.0% | Greek |

Polish (1 sample, 100%) is excluded from the tables above due to insufficient sample size.

#### Unsupported languages (54)

The following CommonLID languages have no mapping to a spirit-guess code and are scored as 0:

ace, acf, amh, arg, asm, bak, bcl, bik, bre, cmn, crh, ext, fro, fry, fuv, gaz, gcf, gcr, gla, gle, gom, grc, gug, guw, ibo, jav, jpn, kab, kan, kik, kor, lij, lin, ltg, lug, mlg, msa, nyn, oci, orm, ory, pcm, rcf, san, sna, tam, tat, tuk, vec, wuu, yor, yue, zho, zsm

Notable absences include Chinese (cmn/zho/wuu/yue), Japanese (jpn), Korean (kor), Amharic (amh), Tamil (tam), Yoruba (yor), and Javanese (jav).

### Analysis

**1. Non-Latin script support is broken.** Languages written in Telugu, Hebrew, Thai, Malayalam, Bengali, Devanagari (Marathi), Gurmukhi, Gujarati, and Greek scripts all score 0%. The ngram detector appears to only handle Latin, Cyrillic, and Arabic scripts effectively. Hindi (Devanagari) is the sole exception at 86.3%, suggesting it may be detected via heuristics separate from the n-gram model.

**2. Arabic dialect handling works well.** All six Arabic varieties in CommonLID (Standard, Egyptian, Moroccan, Najdi, Sudanese, Tunisian) are mapped to `ar` and achieve 77-94% accuracy. This is a strength relative to models that attempt fine-grained Arabic dialect identification and get confused.

**3. Uzbek is the largest single failure.** Uzbek (Latin script) has 43,189 samples --- the single largest language in CommonLID --- but spirit-guess achieves only 0.3% accuracy. Predictions scatter across Azerbaijani, Croatian, Estonian, and other languages. Since Uzbek is weighted heavily in the micro average, this alone drops the full-dataset micro accuracy by ~11 percentage points.

**4. Closely related languages cause confusion.** Czech (37.8%) is frequently misidentified as Slovak. Tagalog/Filipino (39-43%) are confused with Cebuano and Indonesian. English (76.0%) is unexpectedly low, likely due to code-mixed web text and short snippets.

**5. Coverage is the primary bottleneck.** Spirit-guess supports 55 of 109 CommonLID languages. The 54 unsupported languages --- including major languages like Chinese, Japanese, Korean, and Amharic --- all contribute 0 to the macro average. Even if spirit-guess achieved perfect accuracy on its 55 supported languages, the macro-averaged score over all 109 languages could not exceed 50.5%.

**6. Comparison to CLD3.** CLD3 covers the same number of languages (55) and achieves a macro F1 of 34.3 (all) / 66.2 (covered). Spirit-guess's macro recall of 31.0 (all) / 61.5 (covered) is in the same ballpark, suggesting similar practical performance for lightweight, offline language identification.

## License

MIT
