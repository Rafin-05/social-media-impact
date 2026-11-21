# Social Media Impact on Student Life — DS422 (ML-I) Lab Website

A single-page website showcasing the DS422 Machine Learning-I lab project:
predicting the overall impact of social media on a student
(**Beneficial / Neutral / Negative**) from usage, sleep, stress,
mental-health and academic data.

Submission & evaluation: **22 November 2026**

## How to open it

Just open **`index.html`** in any modern browser — double-click it, no
server, no internet and no installation needed. The trained SVM is
embedded in the page, so the live prediction form works straight from
the file.

## What is inside

- `index.html` — the whole one-page site (inline CSS/JS, no external
  CDN): dataset overview, preprocessing & EDA, model comparison with
  charts, best-model deep dive, a **live prediction form** that runs
  the real trained SVM in the browser, and a performance section.
- `model.json` — the exported SVM (support vectors, dual coefficients,
  scaler, one-hot categories) used by the page.
- `metrics.json` — all metrics, class reports, confusion matrices and
  dataset statistics, computed by `train.py` from the CSV.
- `model/best_model.pkl` — the best model (SVM, RBF kernel), saved by
  the notebook pipeline and retrained by `train.py`.
- `model/preprocessor.pkl` — the fitted StandardScaler + OneHotEncoder.
- `model/label_encoder.pkl` — Beneficial=0, Negative=1, Neutral=2.
- `model/test_sample.csv` — the first 50 held-out test rows, used to
  verify the browser prediction against sklearn.
- `train.py`, `export_model.py`, `verify_js.py`, `check_js.mjs` —
  the scripts that retrain, export and verify the model (in the
  project folder / ZIP root; see below).
- `requirements.txt` — Python packages for retraining.

## The model (verified)

Dataset: `Social_media_impact_on_life_0_5e7l.csv` — 4,500 students,
16 columns. Target: `Overall_Impact` (Beneficial 3,681 / Neutral 654 /
Negative 165). 80/20 stratified split, random_state=42
(3,600 train / 900 test). Missing values (Stress 46, GPA 85) filled
with the median; 0 duplicates.

| Model | Train acc. | Test acc. | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---:|---:|---:|---:|---:|
| KNN (k=3) | 0.9733 | 0.9322 | 0.8726 | 0.7972 | 0.8299 |
| Decision Tree | 1.0000 | 0.9611 | 0.9284 | 0.9176 | 0.9229 |
| Random Forest (100 trees) | 1.0000 | 0.9667 | 0.9429 | 0.9060 | 0.9237 |
| **SVM (RBF) — best** | **0.9811** | **0.9667** | **0.9226** | **0.9470** | **0.9332** |

The browser prediction is the real SVM: the page implements the
one-vs-one RBF decision function from `best_model.pkl` in plain
JavaScript. It was verified against sklearn on 50 held-out test rows:
**100% agreement** (run `python verify_js.py` to repeat the check).

## How to retrain

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py        # retrains all 4 models, saves model/*.pkl + metrics.json
                                 # (fails if the SVM does not match the notebook within 0.002)
.venv/bin/python export_model.py # exports model.json and injects it + metrics.json into index.html
.venv/bin/python verify_js.py    # checks the browser SVM against sklearn (must be 100%)
```

Note: the retrain reproduces the notebook exactly with
`scikit-learn==1.6.1` (pinned in `requirements.txt`). With
scikit-learn 1.9 the Random Forest numbers shift slightly
(test accuracy 0.9611 instead of 0.9667), because the forest
implementation changed between versions — the SVM, which is the
model used on the site, is identical either way.

## Honest limits

- Classes are imbalanced (81.8% Beneficial), so macro F1 — not
  accuracy — is the fair comparison; the site says this plainly.
- Decision Tree / Random Forest reach 1.0000 training accuracy,
  suggesting some overfitting.
- No cross-validation or hyperparameter tuning was done in the
  original notebook; results come from a single 80/20 split.
- The dataset source was not stated beyond the supplied course
  file, so no Kaggle/UCI link is claimed on the site.
- The prediction is a lab demo, not a medical or psychological
  diagnosis.
