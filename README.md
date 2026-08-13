# Brain Age Prediction and MRI Site Harmonization

This repository contains the implementation of a machine learning pipeline for predicting
**brain age** from regional cortical thickness (CT) and fractal dimension (FD) features
extracted from T1-weighted brain MRI, and for studying how much **inter-site/scanner effects**
bias this prediction, and how much of that bias is recovered by **ComBat-GAM harmonization**.
Developed as the final project for the *AI for Medicine* course (Prof. Stefano Diciotti,
University of Bologna).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21904829.svg)](https://doi.org/10.5281/zenodo.21904829)

**Abstract.** Pooling brain MRI across scanners and institutions introduces systematic,
non-biological variance ("site effects") that can confound downstream machine learning models.
Using a 36-site, 1,740-subject multicenter dataset of cortical thickness and fractal dimension
features, we first show that a classifier can identify a subject's acquisition site from these
features alone, far above chance — direct evidence of a site effect. We then train
Leave-One-Site-Out cross-validated regression models (ElasticNet, Random Forest, XGBoost) to
predict chronological age, before and after ComBat-GAM harmonization, and quantify how much
harmonization improves cross-site generalization (MAE, RMSE, pooled R², paired significance
testing) and how much it reduces site separability. We document, transparently, an
architectural limitation of ComBat we encountered along the way — it cannot harmonize a
genuinely unseen site out-of-sample — and the standard workaround adopted instead.

Built on top of data released by:

> Marzi, C., Giannelli, M., Barucci, A., Tessa, C., Mascalchi, M., Diciotti, S.
> *Efficacy of MRI data harmonization in the age of machine learning: a multicenter study
> across 36 datasets.* Scientific Data (2024). https://doi.org/10.1038/s41597-023-02421-6

## Index

- [License](#license)
- [Data](#data)
- [Installation](#installation)
- [Running the pipeline](#running-the-pipeline)
- [Inference](#inference)
- [Configuration](#configuration)
- [Methodology summary](#methodology-summary)
- [Reproducibility (FAIR)](#reproducibility-fair)
- [Ethics and data privacy](#ethics-and-data-privacy)
- [Citation](#citation)

## License

The code in this repository is released under the MIT license — see [LICENSE](LICENSE).
The underlying neuroimaging datasets (Marzi & Diciotti, Zenodo records 7845311 / 7845361) are
distributed by their original authors under CC BY(-NC)-SA 3.0 — see the Zenodo records for
exact terms; academic/research use only.

## Data

The notebook downloads two CSVs directly from Zenodo at runtime — **data are not
redistributed** in this repository.

| Part | Zenodo record | File | Subjects | Sites |
|------|----------------|------|----------|-------|
| Part I  | https://zenodo.org/records/7845311 | `multicenter_CT-FD_features_1.csv` | 1,189 | ABIDE I/II, FCP-ICBM, CoRR-NKI2 |
| Part II | https://zenodo.org/records/7845361 | `multicenter_CT-FD_features_2.csv` | 551 | IXI |

Each row is one subject (one T1-weighted scan): `Subject`, `SITE`, `Age`, `Sex` (0 = male,
1 = female), plus 22 regional CT/FD columns (whole cortex, left/right, 4 lobes × 2 hemispheres,
for both cortical thickness and fractal dimension). No preprocessing beyond merging the two
parts and column-based feature selection is required — the features are already extracted,
region-level summary statistics, not raw images.

## Installation

No local installation is required — the notebook is self-contained and installs its own
dependencies in its first cell when run in Google Colab. To reproduce the environment locally
instead:

```bash
git clone https://github.com/<your-username>/AI-for-Medicine-Project.git
cd AI-for-Medicine-Project
pip install -r requirements.txt
jupyter notebook notebooks/AI_for_Medicine_BrainAge.ipynb
```

`requirements.txt` pins the key package versions (`neuroHarmonize`, `neuroCombat`, `xgboost`,
`scikit-learn`, etc.) used to produce the results in the report.

Repository layout:

```
notebooks/
  AI_for_Medicine_BrainAge.ipynb   # main Colab notebook (self-contained, run top to bottom)
results/                           # cached CSVs + figures (populated after a full run)
report/
  (final PDF report, from the official course template)
requirements.txt                   # pinned dependency versions (for reference / local runs)
LICENSE                            # MIT license for the code in this repository
```

## Running the pipeline

1. Open `notebooks/AI_for_Medicine_BrainAge.ipynb` in Google Colab
   (`File > Upload notebook`, or via the GitHub integration).
2. Run all cells top to bottom (`Runtime > Run all`). The first cells install dependencies and
   download the data directly from Zenodo — no manual download needed.
3. All models (ElasticNet, Random Forest, XGBoost) are deterministic given a fixed seed — no
   GPU is required and none of the non-determinism concerns that apply to deep learning models
   apply here (see [Reproducibility](#reproducibility-fair)).
4. Leave-One-Site-Out cross-validation with nested hyperparameter search (36 sites × 3 models ×
   2 harmonization conditions) is the most expensive part — see
   [Configuration](#configuration) to control it.

## Inference

Two things are demonstrated on **genuinely held-out** predictions — i.e., made by a model that
never saw that subject's site during training, not just that subject:

- A sample of out-of-fold Leave-One-Site-Out predictions (age prediction), pulled directly from
  the cross-validation results rather than from a separately fit "final" model — this avoids
  re-introducing the very train/test ambiguity the cross-validation was designed to avoid.
- Section 8 of the notebook documents, explicitly, why a fresh out-of-sample apply of ComBat to
  a brand-new site is not used for this: `neuroHarmonize`'s `harmonizationApply` supports adding
  new subjects to an *already-known* site/batch, not a *genuinely unseen* one.

## Configuration

The notebook has no external config file — the equivalent "knobs" are constants set near the
top (Section 1) and Section 6:

| Constant | Default | Meaning |
|---|---|---|
| `RANDOM_STATE` | `42` | Global seed for NumPy, all model constructors, all CV splitters. |
| `RUN_FULL_TRAINING` | `True` | `True` re-runs the full nested LOSO-CV from scratch; `False` loads cached CSVs from `results/` instead (fast walkthrough). |
| `N_ITER_SEARCH` | `20` | Number of `RandomizedSearchCV` draws per model, per outer LOSO fold (inner hyperparameter tuning). |
| `INNER_CV` | `KFold(n_splits=3)` | Inner cross-validation splitter used for hyperparameter search inside each training fold. |

## Methodology summary

1. Merge Part I + Part II, exploratory analysis of site/age distributions.
2. Quantify the site effect: a classifier predicting acquisition `SITE` from raw CT/FD
   features.
3. Leave-One-Site-Out cross-validation (LOSO-CV) for age prediction, **before** harmonization.
4. ComBat-GAM harmonization (`neuroHarmonize`, `smooth_terms=["Age"]`), fit **once, globally**
   on the full cohort — a genuinely out-of-sample, per-fold application is not supported by
   ComBat's own math for a site unseen during fitting (see notebook Section 8 for the concrete
   failure this caused and the documented trade-off of the global-fit workaround used instead).
   The non-linear (GAM) covariate term was needed: standard linear-covariate ComBat measurably
   *hurt* prediction performance here, consistent with the known non-linear CT/FD–age
   relationship.
5. Repeat LOSO-CV for age prediction **after** harmonization; compare MAE/RMSE/pooled R²
   (per-site R² is reported too, with a note on why it can be misleadingly negative for
   narrow-age-range sites — see notebook Section 7), plus a paired Wilcoxon test and a
   per-site win/loss count.
6. Repeat the site classifier after harmonization to show the reduction in site separability.
7. Feature importance / model interpretation, brain-age gap discussion, held-out inference
   demo.

## Reproducibility (FAIR)

- **Findable**: persistent identifier — [10.5281/zenodo.21904829](https://doi.org/10.5281/zenodo.21904829)
  (Zenodo archive of release `v1.0.0`).
- **Accessible**: this GitHub repository (code) + Zenodo (permanent archive), MIT license.
- **Interoperable**: standard Python/Jupyter, dependencies pinned in `requirements.txt`.
- **Reusable**: documented methodology, README, and inline notebook markdown explaining every
  design decision (including the ones that didn't work — see notebook Section 8).
- All randomness seeded; no GPU-induced non-determinism (no deep learning models used).

## Ethics and data privacy

All data used are secondary, publicly available, de-identified datasets (no raw images, only
aggregated regional morphometric features), originally collected under informed consent and
institutional ethics approval by the source studies (ABIDE, FCP, CoRR, IXI). This project is
for academic/research purposes only and makes no clinical claims. See the final report,
Section 11, for the full discussion.

## Citation

If you use this repository, please cite it via its Zenodo DOI:
[10.5281/zenodo.21904829](https://doi.org/10.5281/zenodo.21904829) (see the "Cite as" section
on the Zenodo record for BibTeX/APA/other formats), and cite the underlying dataset:

> Marzi, C., Giannelli, M., Barucci, A., Tessa, C., Mascalchi, M., Diciotti, S.
> *Efficacy of MRI data harmonization in the age of machine learning: a multicenter study
> across 36 datasets.* Scientific Data (2024). https://doi.org/10.1038/s41597-023-02421-6
