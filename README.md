# flare-forecast

Forecasts near-term IBD disease activity from longitudinal gut microbiome composition, validated subject by subject against real clinical severity scores from the HMP2/IBDMDB cohort. Includes an honest check of whether the microbiome signal is real or just a well-calibrated guess.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)

---

## Overview

Most microbiome-IBD analysis is cross-sectional: find taxa that correlate with disease activity right now. That's a different, easier question than the clinically useful one, which is whether a patient's current gut microbiome tells you anything about where their disease is heading in the next few weeks. This project goes after that harder question directly: given microbiome composition and clinical score at timepoint t, forecast the clinical score 2-4 weeks later.

Two problems that a naive version of this analysis gets wrong, and that this project tries to build around:

- **Repeated-measures leakage.** HMP2 subjects have up to 24 timepoints each. A random train/test split lets a patient's near-identical microbiome from adjacent weeks land in both folds, which inflates accuracy on a metric that isn't really measuring generalization to a new patient. Every model here uses subject-grouped cross-validation (`GroupKFold` or `LeaveOneGroupOut` on `Participant ID`), never a random split.
- **Uncredited baselines.** A model that combines microbiome plus current score can beat "predict no change" purely by learning a slope and intercept on the current score, with the microbiome contributing nothing. That's actually what happened during development here (see Validation/Results below). It got caught by adding a fitted score-only baseline and auditing the model's coefficients with SHAP, instead of just reporting the flattering comparison and moving on.

## Features

- `scripts/download_data.py` downloads the merged HMP2/IBDMDB cohort tables (clinical metadata, MetaPhlAn species-level taxonomy, HUMAnN pathway/EC functional profiles, metabolomics BIOM) from ibdmdb.org's Globus-backed file store. There's no public API or manifest for this, so the URLs were found by hand-scraping the site's product pages, then verified with `HEAD` requests before being hardcoded.
- `scripts/eda.py` establishes real feature and sample counts before any modeling, and resolves the open single- vs multi-omic question with actual numbers instead of assumption: 578 species-level taxa, 22,113 pathways, and 167,854 ECs against 130 subjects (regularization is mandatory either way), and only 30% of metagenomics timepoints (473/1585) have a same-week paired metabolomics sample. So metabolomics fusion got deferred rather than cutting the modeling set by 70% up front.
- `src/flare_forecast/data.py` builds diagnosis-specific datasets around a fact confirmed empirically: HBI and SCCAI are different instruments scored for different diagnoses (689/693 non-null HBI rows are Crohn's, 436/452 non-null SCCAI rows are ulcerative colitis). `build_baseline_dataset` (same-timepoint) and `build_forecast_dataset` (2-4-week-ahead (t, t+1) pairs, built from every same-subject timepoint gap in that window, not just consecutive visits, since HMP2's sampling is irregular) never pool the two scores onto one shared scale.
- `scripts/train_forecast.py` compares six predictors under leave-one-subject-out CV: naive persistence, a fitted score-only linear regression, ElasticNet on raw species abundance, ElasticNet on species plus score, and both again on `ecology.py`'s low-dimensional summary features. The last two exist because the species-only combined model turned out not to be using the microbiome at all.
- `scripts/shap_plausibility.py` checks what the "combined" ElasticNet actually learned: SHAP values (`shap.LinearExplainer`) plus a direct nonzero-coefficient count, checked against a reference list of IBD-associated taxa from the literature (Enterobacteriaceae and *R. gnavus* overrepresentation, depleted butyrate producers, per Lloyd-Price et al. 2019). It found that the cross-validated hyperparameter search prefers pure-Lasso regularization strong enough to zero out nearly every species coefficient, 0 out of 123 nonzero for Crohn's and 2 out of 121 for UC.
- `src/flare_forecast/ecology.py` computes a real dysbiosis score (median Bray-Curtis dissimilarity to a 429-sample, 27-subject non-IBD reference cohort, the same construct Lloyd-Price et al. used, reimplemented here since per-sample scores aren't in the raw downloads), plus Shannon diversity and species richness, as a roughly 7-feature alternative to 578 raw species competing for signal at 51-83 training subjects.
- Every model comparison and the SHAP audit results get written to `results/forecast.json` so they can be checked later instead of just printed and lost.

## Tech Stack

**Data / modeling:** Python 3.12, pandas, NumPy, SciPy, scikit-learn (`ElasticNet`, `GridSearchCV`, `GroupKFold`/`LeaveOneGroupOut`), SHAP
**Data acquisition:** `requests`, hand-verified Globus-backed download URLs
**Storage:** flat files. Gzipped TSV/CSV in `data/raw/` (gitignored, about 281MB, fetched on demand), JSON results in `results/`
**Not yet built:** a frontend or deployment layer (see Roadmap)

## Installation

```bash
git clone https://github.com/saishettar/flare-forecast.git
cd flare-forecast

python -m venv .venv
.venv\Scripts\activate        # Windows; `source .venv/bin/activate` on Linux/Mac
pip install -r requirements.txt

python scripts/download_data.py   # ~281MB: metadata, taxonomy, pathways, ECs, metabolomics
```

## Usage

```bash
python scripts/eda.py
```

```
=== Sample / subject counts ===
metagenomics: 1638 samples, 130 subjects
metabolomics: 546 samples, 106 subjects

=== Clinical activity score coverage ===
hbi: 2145 non-null rows
sccai: 1335 non-null rows

=== Paired MGX/MBX timepoints (same subject, same week_num) ===
paired timepoints: 473 / 1585 MGX timepoints (30%)
subjects with >=1 paired timepoint: 106

=== Feature dimensionality vs. sample count ===
taxonomy: 932 features x 1638 samples
pathways: 22113 features
ECs: 167854 features
```

Then, to run the full forecasting comparison (about 20-25 minutes: leave-one-subject-out CV with a nested hyperparameter search per fold, across six models and two diagnoses):

```bash
python scripts/train_forecast.py
```

Results land in `results/forecast.json`. See Validation/Results below for the actual numbers.

## Project Structure

```
flare-forecast/
├── scripts/
│   ├── download_data.py       # Pulls merged HMP2/IBDMDB tables from ibdmdb.org's Globus store
│   ├── eda.py                 # Feature/sample-count audit; resolves the single- vs multi-omic call
│   ├── train_baseline.py      # Same-timepoint HBI/SCCAI regression (cross-sectional baseline)
│   ├── train_forecast.py      # 2-4wk-ahead forecast: persistence vs. score-only vs. species vs. ecology
│   └── shap_plausibility.py   # SHAP + coefficient audit of what the "combined" model actually learned
├── src/flare_forecast/
│   ├── data.py                 # Metadata/taxonomy loading, dataset builders, (t, t+1) pair construction
│   ├── features.py             # PrevalenceFilter, ArcsinSqrtTransform (compositional preprocessing)
│   └── ecology.py              # Shannon diversity, richness, Bray-Curtis dysbiosis score
├── results/
│   └── forecast.json           # LOSO R^2/MAE per model, written by train_forecast.py
├── data/
│   ├── raw/                    # Downloaded HMP2 source tables (gitignored, ~281MB)
│   └── processed/              # Reserved for derived artifacts (unused so far)
└── requirements.txt
```

## Validation / Results

**Same-timepoint baseline** (`train_baseline.py`, subject-grouped nested CV, `GroupKFold(5)`/`GroupKFold(4)`): cross-sectional microbiome composition alone is a weak predictor of same-day clinical activity score. CD mean R² is about -0.09, UC mean R² is about -0.06, both consistent with zero or worse within one standard deviation. This is reported as-is rather than tuned until it looked better. It's a known-hard task in the field, and it de-risked the modeling approach before adding the temporal element.

**Forecasting** (`train_forecast.py`, leave-one-subject-out CV, 714 CD and 474 UC (t, t+1) pairs 2-4 weeks apart):

| model | CD R² | UC R² |
|---|---|---|
| persistence (y = score_t) | 0.193 | 0.256 |
| score_regression (fitted, no microbiome) | **0.328** | **0.385** |
| microbiome only (578 species) | -0.036 | -0.144 |
| combined (species + score_t) | 0.299 | 0.327 |
| ecology only (diversity/richness/dysbiosis) | -0.045 | -0.111 |
| ecology_combined (ecology + score_t) | 0.322 | 0.346 |

Neither raw species composition nor derived ecological state (diversity, richness, a Bray-Curtis dysbiosis score, and their deltas from the prior visit) improves 2-4-week activity forecasts beyond a plain fitted regression on today's score alone, in this cohort, under leave-one-subject-out validation. Both microbiome representations underperform persistence on their own, and both "combined" variants land at or slightly below `score_regression`, never clearly above it.

That wasn't the first framing tried. An earlier run reported "combined beats naive persistence" (0.300 vs. 0.194), which looked like real microbiome signal until a SHAP audit (`shap_plausibility.py`) showed the fitted model's species coefficients were nearly all zero (0/123 for CD, 2/121 for UC). The gain was really just a learned slope and intercept on score_t that naive persistence, which forces slope=1, doesn't get credit for. `score_regression` was added specifically to make that comparison fair. The ecology feature set came afterward, built on the idea that 578 raw species might just be too high-dimensional for 51-83 subjects to find signal in even if it exists. Checking its fitted coefficients directly shows the same pattern: mostly zero, small weight on one to three ecology features, dominant weight on score_t (for CD, a score_t coefficient of 1.45 against a dysbiosis_t of -0.08 and everything else exactly zero; for UC, score_t at 1.20 against three small nonzero ecology terms).

A few caveats that apply regardless of the exact numbers above. 51-83 training subjects per diagnosis means every LOSO fold is a substantial share of the data, so reported R² has real variance. HBI and SCCAI are noisy self-report clinical instruments, not lab measurements, which caps how predictable they can be from any input. And the source paper for this cohort never reports a forecasting R² to compare against, since it's a cross-sectional analysis, so there isn't a strong external number for what "good" looks like here.

## Architecture

```
ibdmdb.org (Globus-backed store, no public API/manifest)
      |  download_data.py -- hand-verified URLs, HEAD-checked before hardcoding
      v
data/raw/  --  hmp2_metadata_2018-08-20.csv, taxonomic_profiles_3.tsv.gz, ...
      |
      |  data.py: load_metadata() / load_species_taxonomy()
      v
species-level relative abundance (578 taxa) x clinical metadata (HBI/SCCAI, week_num, Participant ID)
      |
      +-- build_baseline_dataset -------------> train_baseline.py
      |                                          (same-timepoint ElasticNet, GroupKFold(5))
      |
      +-- build_forecast_dataset(_ecology) ---> train_forecast.py
                                                   |  6 models x LeaveOneGroupOut(Participant ID)
                                                   v
                                             results/forecast.json
                                                   |
                                                   v
                                        shap_plausibility.py
                                        (single full-data fit; SHAP values +
                                         nonzero-coefficient audit of "combined")
```

## Prior Art

The source dataset and its original analysis are [Lloyd-Price et al. 2019, *Nature*](https://www.nature.com/articles/s41586-019-1237-9) (the HMP2/IBDMDB paper) and its companion repo, [biobakery/hmp2_analysis](https://github.com/biobakery/hmp2_analysis). That paper's own dysbiosis-scoring approach, median Bray-Curtis dissimilarity to a non-IBD reference cohort, is what `ecology.py`'s `dysbiosis_score` reimplements from scratch, since per-sample scores aren't included in the raw downloads and the analysis repo wasn't pulled in as a dependency. The original paper is cross-sectional dysbiosis characterization, not forecasting. The leave-one-subject-out (t, t+1) forecasting task here, and specifically the score-only-baseline and SHAP audit built to check whether "combined beats persistence" claims are real, are this project's own additions rather than reproductions of established methodology. Most of the modeling itself, ElasticNet with nested subject-grouped CV and arcsin-sqrt-transformed relative abundance, follows standard, well-established microbiome ML practice rather than inventing anything new there.

## Roadmap / Limitations

**Gaps closed during development:**
- The single- vs multi-omic decision was resolved with real feature/sample counts in `eda.py` rather than assumed at the start.
- An early "combined model beats persistence" result looked like a real finding. A SHAP audit caught that the model's species coefficients were nearly all zero, so a fitted score-only baseline got added to make the comparison honest.
- Tried low-dimensional ecological summary features (diversity, richness, dysbiosis score, trajectory deltas) as an alternative to 578 raw species, on the idea that dimensionality itself was hiding real signal. It wasn't. The ecology-based models show the same pattern, mostly-zero fitted coefficients and no improvement over `score_regression`, which is evidence the earlier negative result is real rather than an artifact of one particular feature representation.
- Building the dysbiosis score turned up a real QC issue: 11 of 1638 metagenomics samples had exactly 0 abundance across all 578 species, which turned out to be failed profiling runs rather than genuine zero-diversity samples, and they'd been silently included in every model up to that point. Fixed by dropping them at the source in `load_species_taxonomy`. The shift in reported numbers was small (a handful of contaminated rows in an already-large sample) but real.
- The first LOSO run used `GridSearchCV(n_jobs=-1)` inside the outer loop, which respawns a joblib worker pool per call, 164 calls total. That's fine on Linux but effectively hung on Windows process-spawn overhead. Fixed by switching to `n_jobs=1`.

**Genuine design decisions, not oversights:**
- Single-omic (metagenomics only) for all models here. Metabolomics fusion would cut the modeling set by about 70% since only 30% of timepoints have a paired sample, so it's deferred to a future enrichment pass over that smaller subset rather than folded into the primary models.
- Species-level taxonomy only, not the full kingdom-through-species rank stack in the raw file, since using every rank at once double-counts abundance (a phylum's total is the sum of its species).
- HBI and SCCAI are kept diagnosis-separate rather than pooled onto one severity scale, since they're different clinical instruments scored for different diagnoses.

**Stretch goals, not started:**
- A patient-trajectory viewer, showing predicted vs. actual activity score over a held-out subject's real timepoints.
- Metabolomics fusion over the roughly 30% paired-sample subset.
- A second, independent microbiome-IBD cohort for held-out validation. The SHAP-based plausibility check above was used instead, as a cheaper independent sanity check on the same cohort.

## License

[MIT](LICENSE)

## Acknowledgments

- [ibdmdb.org](https://ibdmdb.org) and the HMP2 Consortium, for the source data
- [Lloyd-Price et al. 2019, *Nature*](https://www.nature.com/articles/s41586-019-1237-9) and [biobakery/hmp2_analysis](https://github.com/biobakery/hmp2_analysis), whose dysbiosis-scoring construct this project reimplements
- [SHAP](https://github.com/shap/shap) (Lundberg & Lee), used for the model-plausibility audit
