# Job Market NLP & ML Analyzer

**A data-driven analysis of the LinkedIn job postings market — cleaning ~33,000 postings, then applying three machine-learning tasks to uncover structure in the labor market.**

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange.svg)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg)](https://huggingface.co/datasets/xanderios/linkedin-job-postings)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-lightgrey.svg)](notebooks/Job_Market_Exploration.ipynb)

---

##  Results at a Glance

| Task | Type | Model | Metric | Score |
|---|---|---|---|---|
| **Salary prediction** | Regression | RidgeCV (log-target) | R² / RMSE | **0.645** / **$32,989** |
| **Seniority classification** | Classification | LogisticRegression | Macro-F1 | **0.709** |
| **Skill clustering** | Unsupervised | KMeans (k=10) | 10 coherent tracks | ✅ |

All three models substantially outperform naive baselines. The salary model reduces RMSE by **40%** vs. predicting the mean; the seniority classifier reaches **73.7% accuracy** against a 33% random baseline; and the clustering step recovers **10 interpretable labor-market segments** without any predefined taxonomy.

---

##  Key Findings

1. **Salary is predictable from text alone.** RidgeCV reaches R² = 0.645 with RMSE $32,989 — a 40% reduction over the mean-predictor baseline. Job descriptions carry real wage signal.
2. **The salary signal is semantic, not numeric.** ~60% of descriptions contain an explicit salary range that would leak the target. We strip these numeric tokens before vectorization — the model loses only **ΔR² = 0.009**, proving it learned domain language (seniority markers, skill mentions, experience phrases), not answer leakage.
3. **Linear models dominate on sparse TF-IDF.** Ridge beats RandomForest (0.645 vs. 0.537 R²) — the expected outcome for high-dimensional sparse text, where axis-aligned tree splits can't efficiently exploit the representation.
4. **Seniority is distinguishable from language.** Macro-F1 = 0.709 with LogisticRegression. Entry and Mid-Senior classify reliably (F1 0.77 and 0.72); Director is the bottleneck (F1 0.63) due to class imbalance (6.9%) and vocabulary overlap with senior ICs.
5. **Class imbalance trades accuracy for macro-F1.** RandomForest achieves higher raw accuracy (75.4% vs. 73.7%) but lower macro-F1 (0.685 vs. 0.709). RF adopts a precision-heavy Director strategy (P=0.94, R=0.36) that starves recall — a classic imbalance artifact. LR's balanced treatment is preferable here.
6. **The labor market has 10 recoverable sectors.** KMeans cleanly separates office/insurance, software, marketing, manufacturing, clinical, sales, construction, accounting, retail, and travel nursing — **including a fine-grained distinction between hospital clinical work and agency-placed travel nursing contracts**, discovered without any predefined labels.

---

##  Methodology

### Data

- **Source**: [xanderios/linkedin-job-postings](https://huggingface.co/datasets/xanderios/linkedin-job-postings) (Hugging Face)
- **Size**: 33,241 cleaned postings
- **Salary coverage**: 11,003 rows (33.1%) with valid annual salary data — 11,002 after the `<50-char` description filter
- **Seniority coverage**: 24,065 rows with labeled seniority — 24,063 after the `<50-char` filter
- **Target distribution** (salary regression subset): mean $100,177, median $89,500, std $53,661

### Pipeline

```
data/raw/job_postings.csv (28 columns)
        │
        ▼
[src/nlp_cleaner.py]  ── HTML strip, lowercase, regex preserve C++/C#
        │
        ▼
data/processed/job_postings_cleaned.csv (9 columns)
        │
        ▼
[src/analysis.py]  ── annualize salary, map 7→3 seniority classes, flag salary outliers
        │
        ▼
data/processed/job_postings_ml_ready.csv (12 columns)
        │
        ├──────────────────┬────────────────────┬─────────────────────┐
        ▼                  ▼                    ▼                     ▼
[ml_models.py]     [ml_models.py]     [ml_models.py]         [visualization.py]
  --task salary      --task seniority   --task cluster          polished charts
        │                  │                    │                     │
        ▼                  ▼                    ▼                     ▼
reports/*.json + models/*.joblib                          visualizations/*.png
        │
        ▼
[notebooks/Job_Market_Exploration.ipynb]  ── final portfolio narrative
```

### Key design decisions

- **Log-transformed salary target** — salaries are right-skewed; `log1p` stabilizes variance for linear regression. Metrics are back-transformed to dollars for interpretability.
- **Leakage guard** — a regex strips numeric salary ranges from `description_clean` **before** TF-IDF vectorization (the source data remains intact for display). This is the single most important methodological step for honest R².
- **`is_salary_valid` flag** — outlier rows (`<$15k` or `>$500k` after annualization) are marked in the CSV rather than silently dropped, so all downstream tasks use a single source of truth for filtering.
- **`class_weight='balanced'`** on classifiers — mandatory given Director is only 6.9% of labeled data.
- **Macro-F1, not accuracy** — the correct evaluation metric for imbalanced multi-class problems.
- **Held-out 20% test split** for both regression and classification, stratified for the classifier.

---

##  Results

### 1. Salary Regression

<img src="visualizations/salary_by_seniority.png" alt="Salary by seniority" width="720">

**Salary is monotonic with seniority** — median annual salary climbs from ~$65k (Entry) → ~$110k (Mid-Senior) → ~$157k (Director), validating the annualization approach. Full distribution:

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Baseline (predict test mean) | $55,382 | $42,933 | 0.000 |
| RandomForest (tuned) | $37,690 | $24,041 | 0.537 |
| **RidgeCV (final)** | **$32,989** | **$20,777** | **0.645** |

The Ridge vs. RandomForest gap (0.645 vs. 0.537) is expected for TF-IDF text — sparse high-dimensional inputs favor L2-regularized linear models over axis-aligned tree splits.

**Leakage ablation**: an initial run *without* the salary-token filter achieved R² = 0.654. Stripping salary numbers cost only **ΔR² = 0.009**, proving the model learned semantic content rather than answer leakage.

<img src="visualizations/salary_pred_vs_actual.png" alt="Predicted vs actual salary" width="560">

The 2D hexbin of RidgeCV test-set predictions shows mass concentrating along the diagonal — the model captures the trend but under-predicts the extremes, as expected for text-only regression.

### 2. Seniority Classification

<img src="visualizations/seniority_distribution.png" alt="Seniority distribution" width="720">

Mid-Senior (11,258) and Entry (11,149) dominate; Director (1,658) is the minority class at 6.9% — hence the mandatory `class_weight='balanced'`.

| Model | Accuracy | Macro-F1 | Director F1 |
|---|---|---|---|
| **LogisticRegression (final)** | **0.7370** | **0.7089** | **0.634** |
| RandomForest | 0.7540 | 0.6846 | 0.522 |

**Why LR wins on macro-F1** despite lower accuracy: RandomForest adopts a precision-heavy Director strategy (P=0.94, R=0.36), trading recall for confidence. LR is balanced (P=0.53, R=0.78) — it catches most Directors at the cost of false positives. For macro-F1, balanced treatment wins.

<img src="visualizations/seniority_confusion_matrix.png" alt="Seniority confusion matrix" width="520">

Row-normalized color with raw counts as cells. Director is the weakest class (off-diagonal bleed into Mid-Senior), consistent with its imbalance and vocabulary overlap.

### 3. Skill Clustering

<img src="visualizations/cluster_sizes.png" alt="Cluster sizes" width="720">

KMeans on TF-IDF (5,000 features, unigram) recovered **10 interpretable labor-market segments** without any predefined taxonomy:

| Track | Size | Marker terms (from top-10 TF-IDF) |
|---|---|---|
| Office / insurance services | 9,509 | service, office, benefits, clients, insurance |
| Software / data engineering | 4,979 | software, technical, engineering, systems, cloud, security |
| Marketing / product | 4,417 | marketing, product, media, content, teams |
| Manufacturing / industrial | 4,056 | equipment, safety, maintenance, manufacturing, production, quality |
| Hospital clinical / nursing | 2,553 | patient, nursing, hospital, clinical, healthcare |
| Sales / business dev | 2,413 | sales, territory, customers, selling, relationships |
| Construction / project mgmt | 1,489 | project, construction, engineering, cost, manager |
| Accounting / finance | 1,477 | accounting, tax, audit, cpa, financial |
| Retail store ops | 1,402 | store, merchandise, associates, retail, merchandising |
| Travel nursing agency | 911 | carestaff, aya, rn, assignment, travel |

**Fine-grained discovery**: *Hospital clinical* and *Travel nursing agency* are both healthcare, yet KMeans separated them cleanly. The first is hospital-employed clinical work (`patient`, `clinical`); the second is agency-placed travel contracts (`carestaff`, `aya` — actual agency names). Unsupervised clustering recovered **sub-specialty structure within sectors**, which is the strongest evidence that the segment structure is real rather than an artifact.

### 4. Top Skills Across All Postings

<img src="visualizations/top_skills.png" alt="Top skills" width="720">

Color-coded by category (Languages / Data-ML / Cloud-DevOps / Web). `excel` dominates at ~16% of postings but is inflated by verb usages — see Limitations.

---

##  Repository Structure

```
Job_Market_Analyzer/
│
├── data/
│   ├── raw/                          # Downloaded Hugging Face dataset (not committed)
│   └── processed/
│       ├── job_postings_cleaned.csv  # After nlp_cleaner.py
│       └── job_postings_ml_ready.csv # After analysis.py (12 cols, with flags)
│
├── src/
│   ├── nlp_cleaner.py                # HTML strip, text normalization
│   ├── analysis.py                   # EDA, salary annualization, seniority mapping
│   ├── ml_models.py                  # Regression, classification, clustering
│   └── visualization.py              # Polished, portfolio-grade charts
│
├── notebooks/
│   └── Job_Market_Exploration.ipynb  # Final narrative presentation
│
├── models/                           # Trained .joblib artifacts
├── reports/                          # JSON metrics + prediction CSVs
├── visualizations/                   # Portfolio-ready PNGs
│
├── requirements.txt
└── README.md
```

---

##  How to Run

### Prerequisites

- Python 3.10
- ~2 GB disk space
- ~10 minutes total runtime for the full pipeline

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-handle>/Job_Market_Analyzer.git
cd Job_Market_Analyzer

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the raw dataset in data/raw/
#    Download from: https://huggingface.co/datasets/xanderios/linkedin-job-postings
```

### Run the pipeline

```bash
python src/nlp_cleaner.py                    # ~10 sec   → cleaned CSV
python src/analysis.py                       # ~60 sec   → ML-ready CSV + EDA charts
python src/ml_models.py --task salary        # ~5 min    → salary regression
python src/ml_models.py --task seniority     # ~40 sec   → seniority classification
python src/ml_models.py --task cluster       # ~90 sec   → KMeans clustering
python src/visualization.py                  # ~60 sec   → polished portfolio charts
```

Then open the notebook:

```bash
jupyter notebook notebooks/Job_Market_Exploration.ipynb
```

The notebook reads from saved artifacts (`reports/*.json`, `visualizations/*.png`) — it opens in ~5 seconds without re-training.

---

##  Limitations

- **Dataset subset**: 33,241 postings, a subset of the full xanderios corpus (~124k). Salary regression trains on only ~11k rows with valid salary data (33%).
- **Class imbalance**: Director is only 6.9% of labeled data. Even with `class_weight='balanced'`, recall remains the limiting factor.
- **Self-referential titles**: Descriptions frequently repeat the job title in the body ("We are seeking a Senior X..."), which likely boosts seniority classification. Legitimate at prediction time, but makes the task easier than pure content analysis.
- **Salary unit assumptions**: Hourly-to-annual conversion assumes 2,080 hours/year. Standard, but ignores part-time and contract variation.
- **Location not used**: Postings span the US and internationally; salary varies dramatically by geography. Location was not a feature.
- **`excel` skill count inflated**: `\bexcel\b` matches both the software and the verb ("excel at multitasking"). True Excel-skill postings are likely 20–30% lower than the reported 5,357.
- **Clustering** silhouette values (0.011–0.016) are within the expected range for KMeans on sparse TF-IDF but do not indicate a clean global optimum. Best k=10 occurred at the edge of the search range (k=4..10); the choice is validated by semantic coherence, not silhouette.

##  Future Work

- **Location features** — add state/city one-hot features; expected to substantially improve salary R².
- **Transformer embeddings** — a fine-tuned BERT would likely beat TF-IDF on both regression and classification, especially for the Director class where semantics matter more than vocabulary.
- **Deep clustering** — UMAP + HDBSCAN could recover non-convex cluster shapes and identify *more than k=10* segments automatically.
- **Temporal analysis** — if posting dates were available, wage trends by quarter and skill could be tracked.

---

##  Tech Stack

| Layer | Tools |
|---|---|
| **Language** | Python 3.10 |
| **Data** | pandas, NumPy |
| **ML** | scikit-learn (RidgeCV, RandomForest, LogisticRegression, KMeans) |
| **NLP** | TF-IDF vectorization (scikit-learn), BeautifulSoup (HTML cleaning) |
| **Viz** | matplotlib, seaborn |
| **Notebook** | Jupyter |

---

##  What's in the Notebook

[`notebooks/Job_Market_Exploration.ipynb`](notebooks/Job_Market_Exploration.ipynb) walks through the entire analysis in 5 minutes of reading:

1. **Dataset & EDA** — shape, coverage, seniority distribution, top skills
2. **Salary Regression** — model comparison, leakage ablation, honest R² narrative
3. **Seniority Classification** — per-class F1, confusion matrix, why LR beats RF on macro-F1
4. **Skill Clustering** — 10 recovered career tracks, with the fine-grained healthcare distinction
5. **Key Findings** — condensed 6-point summary
6. **Limitations & Future Work** — transparent scope and natural extensions

---

## 👤 Author

**Anandhu Suresh** — BCA Graduate, 2026
[GitHub](https://github.com/anandhusuresh04) · [LinkedIn](https://www.linkedin.com/in/anandhu-suresh-in)

---

*Built as a portfolio project to demonstrate end-to-end data science: cleaning → EDA → modeling → honest reporting.*