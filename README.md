# Brain Age Prediction and MRI Site Harmonization

**AI for Medicine — Final Project (Prof. Stefano Diciotti, University of Bologna)**

## Overview

This project predicts chronological **brain age** from regional cortical thickness (CT) and
fractal dimension (FD) features extracted from T1-weighted brain MRI, and studies how
**inter-site/scanner effects** bias this prediction — and how much of that bias is removed by
**ComBat harmonization** applied out-of-sample.

Data: 1,740 subjects across 36 acquisition sites (ABIDE I/II, FCP-ICBM, CoRR-NKI2, IXI),
released by Marzi, Diciotti et al. as part of:

> Marzi, C., Giannelli, M., Barucci, A., Tessa, C., Mascalchi, M., Diciotti, S.
> *Efficacy of MRI data harmonization in the age of machine learning: a multicenter study
> across 36 datasets.* Scientific Data (2024). https://doi.org/10.1038/s41597-023-02421-6

## Data sources

| Part | Zenodo record | File |
|------|----------------|------|
| Part I  | https://zenodo.org/records/7845311 | `multicenter_CT-FD_features_1.csv` |
| Part II | https://zenodo.org/records/7845361 | `multicenter_CT-FD_features_2.csv` |

Data are **not redistributed** in this repository — the notebook downloads them directly
from Zenodo. License: CC BY(-NC)-SA 3.0 (see Zenodo records for exact terms) — academic/
research use only.

## Repository structure

```
notebooks/
  AI_for_Medicine_BrainAge.ipynb   # main Colab notebook (self-contained, run top to bottom)
report/
  (final PDF report, from the official course template)
requirements.txt                   # pinned dependency versions (for reference / local runs)
LICENSE                            # MIT license for the code in this repository
```

## How to run

1. Open `notebooks/AI_for_Medicine_BrainAge.ipynb` in Google Colab
   (`File > Upload notebook` or via the GitHub integration).
2. Run all cells top to bottom. The first cells install dependencies and download the data
   directly from Zenodo — no manual download needed.
3. Random seeds are fixed throughout (`RANDOM_STATE = 42`). All models used (ElasticNet,
   Random Forest, XGBoost) are deterministic given a fixed seed — no GPU non-determinism is
   involved.
4. The notebook has a `RUN_FULL_TRAINING` flag: set to `True` to re-run the full nested
   Leave-One-Site-Out cross-validation from scratch (can take a while with 36 sites × 3
   models); set to `False` to load precomputed results (saved under `results/`) for a fast
   walkthrough.

## Methodology summary

1. Merge Part I + Part II, exploratory analysis of site/age distributions.
2. Quantify the site effect: a classifier predicting acquisition `SITE` from raw CT/FD
   features.
3. Leave-One-Site-Out cross-validation (LOSO-CV) for age prediction, **before** harmonization.
4. ComBat harmonization (`neuroHarmonize`), fit **only on training sites within each LOSO
   fold** and applied out-of-sample to the held-out site — avoiding data leakage.
5. Repeat LOSO-CV for age prediction **after** harmonization; compare MAE/RMSE/R².
6. Repeat the site classifier after harmonization to show the reduction in site separability.
7. Feature importance / model interpretation, brain-age gap discussion.

## Reproducibility (FAIR)

- Code: this GitHub repository, MIT license.
- Persistent identifier: Zenodo DOI (badge added once archived — see repository releases).
- Dependencies pinned in `requirements.txt`.
- All randomness seeded; no GPU-induced non-determinism (no deep learning models used).

## Ethics and data privacy

All data used are secondary, publicly available, de-identified datasets (no raw images, only
aggregated regional morphometric features), originally collected under informed consent and
institutional ethics approval by the source studies (ABIDE, FCP, CoRR, IXI). This project is
for academic/research purposes only and makes no clinical claims. See the final report,
Section 11, for the full discussion.
