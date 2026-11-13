"""
ml_models.py
------------
Three ML tasks on cleaned job postings, all driven by the same TF-IDF pipeline.

1. Salary regression      : predict annual_salary_est from description_clean
                            Baselines: RidgeCV (fast, strong on text)
                            Main     : RandomForestRegressor (RandomizedSearchCV)
                            Target is log-transformed for stability.
                            Salary-number tokens are stripped from input
                            features to prevent target leakage from JDs
                            that publish their own salary range.

2. Seniority classification : predict seniority_class in
                            {Entry, Mid-Senior, Director}
                            Models   : LogisticRegression (baseline)
                                       RandomForestClassifier (main)
                            Both use class_weight='balanced'.

3. Skill clustering       : KMeans on TF-IDF; k chosen by silhouette score.

Inputs : data/processed/job_postings_ml_ready.csv
Outputs: models/*.joblib      (trained models)
         reports/*.json       (metrics)
         visualizations/*.png (confusion matrices, silhouette, importances)

Usage:
    python src/ml_models.py --task salary
    python src/ml_models.py --task seniority
    python src/ml_models.py --task cluster
    python src/ml_models.py --task all
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_READY_PATH = PROJECT_ROOT / "data" / "processed" / "job_postings_ml_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
VIZ_DIR = PROJECT_ROOT / "visualizations"
for d in (MODELS_DIR, REPORTS_DIR, VIZ_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
MIN_DESC_LEN = 50            # drop noise descriptions
SALARY_MIN = 15_000          # sanity filter — must match analysis.py
SALARY_MAX = 500_000
TFIDF_KW = dict(
    max_features=15_000,
    min_df=5,
    max_df=0.85,
    ngram_range=(1, 2),
    stop_words="english",
)

# Strip numeric salary patterns before TF-IDF.
# Job descriptions often contain their own $X–$Y salary range, which
# would leak the regression target into the features. Removing these
# gives an honest generalization estimate.
SALARY_TOKEN_RE = re.compile(
    r'(?:USD\s*|\$\s*)?\d{1,3}(?:,\d{3})+(?:\.\d+)?'   # 103,700 | $103,700.00
    r'|\b\d+(?:\.\d+)?k\b',                             # 100k | 100.5k
    re.IGNORECASE,
)


def _strip_salary_numbers(text: str) -> str:
    """Remove salary-range tokens from a description before vectorization."""
    return SALARY_TOKEN_RE.sub(" ", text)


sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110


# ------------------------------------------------------------------
# 1. Data loading
# ------------------------------------------------------------------
def load_ml_data() -> pd.DataFrame:
    if not ML_READY_PATH.exists():
        raise FileNotFoundError(
            f"{ML_READY_PATH} not found. Run src/analysis.py first."
        )
    df = pd.read_csv(ML_READY_PATH)
    before = len(df)
    df = df[df["description_clean"].str.len() >= MIN_DESC_LEN].copy()
    print(f"Loaded {len(df):,} rows (dropped {before - len(df):,} with "
          f"description < {MIN_DESC_LEN} chars).")
    return df


def _save_json(payload: dict, name: str) -> None:
    path = REPORTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  -> {path.name}")


# ------------------------------------------------------------------
# 2. Salary regression
# ------------------------------------------------------------------
def run_salary_regression(df: pd.DataFrame) -> None:
    print("\n========== SALARY REGRESSION ==========")
    sub = df.dropna(subset=["annual_salary_est"]).copy()

    # Use the flag written by analysis.py — single source of truth,
    # zero risk of filter drift between analysis and modeling.
    sub = sub[sub["is_salary_valid"]].copy()
    print(f"Regression rows with is_salary_valid=True: {len(sub):,}")

    # Strip salary-number tokens before vectorization (leakage guard).
    stripped_series = sub["description_clean"].apply(_strip_salary_numbers)
    n_stripped = int((stripped_series != sub["description_clean"]).sum())
    print(f"Stripped salary-number patterns from {n_stripped:,} "
          f"of {len(sub):,} descriptions (leakage guard).")
    X_text = stripped_series.values
    y = sub["annual_salary_est"].values

    # Log-transform target: standard for right-skewed salary data.
    # Train on log(1+y); expm1 back at prediction time to report dollars.
    y_log = np.log1p(y)

    (X_train_txt, X_test_txt,
     y_train_log, y_test_log,
     y_train_orig, y_test_orig) = train_test_split(
        X_text, y_log, y, test_size=0.2, random_state=RANDOM_STATE
    )

    print(f"Target (dollars) — train mean=${y_train_orig.mean():,.0f} "
          f"std=${y_train_orig.std():,.0f} | "
          f"test mean=${y_test_orig.mean():,.0f} "
          f"std=${y_test_orig.std():,.0f}")

    vectorizer = TfidfVectorizer(**TFIDF_KW)
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)
    print(f"TF-IDF: {X_train.shape[1]:,} features "
          f"(train {X_train.shape[0]:,} / test {X_test.shape[0]:,})")

    # --- RidgeCV baseline on log-target ---
    print("\n[1/3] RidgeCV baseline (log-target)")
    alphas = [0.01, 0.1, 1, 10, 100, 1000]
    ridge = RidgeCV(alphas=alphas, cv=3)
    ridge.fit(X_train, y_train_log)
    ridge_pred = np.expm1(ridge.predict(X_test))     # back to dollars
    print(f"  Selected alpha: {ridge.alpha_:.4f}")
    ridge_metrics = _reg_metrics(y_test_orig, ridge_pred, "RidgeCV")
    joblib.dump({"vectorizer": vectorizer, "model": ridge},
                MODELS_DIR / "salary_ridge.joblib")

    # --- RandomForest on log-target ---
    print("\n[2/3] RandomForestRegressor (RandomizedSearchCV, log-target)")
    rf = RandomForestRegressor(
        random_state=RANDOM_STATE, n_jobs=-1, n_estimators=100,
    )
    param_dist = {
        "max_depth": [30, 50, None],
        "min_samples_leaf": [2, 5, 10],
        "max_features": [0.05, 0.1],
    }
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=3, cv=2, scoring="r2",
        random_state=RANDOM_STATE, n_jobs=-1, verbose=2,
    )
    search.fit(X_train, y_train_log)
    rf_best = search.best_estimator_
    rf_pred = np.expm1(rf_best.predict(X_test))      # back to dollars
    rf_metrics = _reg_metrics(y_test_orig, rf_pred, "RandomForest (tuned)")
    print(f"  Best params: {search.best_params_}")
    joblib.dump({"vectorizer": vectorizer, "model": rf_best},
                MODELS_DIR / "salary_rf.joblib")

    # --- Baseline: predict train mean (dollars) ---
    mean_pred = np.full_like(y_test_orig, y_train_orig.mean(), dtype=float)
    _reg_metrics(y_test_orig, mean_pred, "Baseline (predict mean)")

    # --- Top RF feature importances ---
    feature_names = vectorizer.get_feature_names_out()
    importances = rf_best.feature_importances_
    top_idx = np.argsort(importances)[-20:][::-1]
    top_features = [(feature_names[i], float(importances[i])) for i in top_idx]

    # --- Save artifacts ---
    pred_df = pd.DataFrame({
        "actual": y_test_orig,
        "ridge_pred": ridge_pred,
        "rf_pred": rf_pred,
    })
    pred_df.to_csv(REPORTS_DIR / "salary_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test_orig, rf_pred, alpha=0.3, s=8)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "r--", lw=1)
    ax.set_xlabel("Actual salary (USD)"); ax.set_ylabel("Predicted")
    ax.set_title("RandomForest — Predicted vs Actual Salary")
    fig.tight_layout(); fig.savefig(VIZ_DIR / "salary_pred_vs_actual.png")
    plt.close(fig)

    _save_json({
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "log_transform": True,
        "leakage_guard": True,
        "n_descriptions_stripped": n_stripped,
        "salary_filter": [SALARY_MIN, SALARY_MAX],
        "ridge_alpha": float(ridge.alpha_),
        "ridge": ridge_metrics,
        "random_forest": rf_metrics,
        "rf_best_params": search.best_params_,
        "top_rf_features": top_features,
    }, "salary_metrics.json")
    print("[3/3] Saved salary artifacts.")


def _reg_metrics(y_true, y_pred, name: str) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"  {name:<24} RMSE=${rmse:>10,.0f}  "
          f"MAE=${mae:>9,.0f}  R²={r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


# ------------------------------------------------------------------
# 3. Seniority classification
# ------------------------------------------------------------------
SENIORITY_LABELS = ["Entry", "Mid-Senior", "Director"]


def run_seniority_classification(df: pd.DataFrame) -> None:
    print("\n========== SENIORITY CLASSIFICATION ==========")
    sub = df.dropna(subset=["seniority_class"]).copy()
    print(f"Classification rows: {len(sub):,}")
    print("Class distribution:")
    print(sub["seniority_class"].value_counts().to_string())

    X_text = sub["description_clean"].values
    y = sub["seniority_class"].values

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(**TFIDF_KW)
    X_train = vectorizer.fit_transform(X_train_txt)
    X_test = vectorizer.transform(X_test_txt)
    print(f"TF-IDF: {X_train.shape[1]:,} features")

    # --- Logistic Regression ---
    print("\n[1/3] LogisticRegression (class_weight='balanced')")
    lr = LogisticRegression(
        max_iter=2000, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_metrics = _clf_metrics(y_test, lr_pred, "LogisticRegression",
                              labels=SENIORITY_LABELS)
    joblib.dump({"vectorizer": vectorizer, "model": lr},
                MODELS_DIR / "seniority_lr.joblib")

    # --- Random Forest ---
    print("\n[2/3] RandomForestClassifier (class_weight='balanced')")
    rf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced_subsample",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = _clf_metrics(y_test, rf_pred, "RandomForest",
                              labels=SENIORITY_LABELS)
    joblib.dump({"vectorizer": vectorizer, "model": rf},
                MODELS_DIR / "seniority_rf.joblib")

    # --- Confusion matrix plot (LR) ---
    cm = confusion_matrix(y_test, lr_pred, labels=SENIORITY_LABELS)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SENIORITY_LABELS, yticklabels=SENIORITY_LABELS,
                ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("LogisticRegression — Confusion Matrix")
    fig.tight_layout(); fig.savefig(VIZ_DIR / "seniority_confusion_matrix.png")
    plt.close(fig)

    _save_json({
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "n_features": int(X_train.shape[1]),
        "logistic_regression": lr_metrics,
        "random_forest": rf_metrics,
        "labels": SENIORITY_LABELS,
    }, "seniority_metrics.json")
    print("[3/3] Saved seniority artifacts.")


def _clf_metrics(y_true, y_pred, name: str, labels: list) -> dict:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    print(f"  {name:<22} Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")
    print(classification_report(y_true, y_pred, labels=labels, digits=4))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "per_class_report": classification_report(
            y_true, y_pred, labels=labels, output_dict=True, digits=4
        ),
    }


# ------------------------------------------------------------------
# 4. Skill clustering
# ------------------------------------------------------------------
def run_clustering(df: pd.DataFrame) -> None:
    print("\n========== SKILL CLUSTERING ==========")
    print(f"Clustering rows: {len(df):,}")

    vectorizer = TfidfVectorizer(
        max_features=5_000, min_df=10, max_df=0.5,
        ngram_range=(1, 1), stop_words="english",
    )
    X = vectorizer.fit_transform(df["description_clean"].values)
    print(f"TF-IDF: {X.shape[1]:,} features")

    # Silhouette sweep
    print("\n[1/3] Silhouette sweep k=4..10")
    scores = {}
    for k in range(4, 11):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        score = float(silhouette_score(X, labels, sample_size=5000,
                                       random_state=RANDOM_STATE))
        scores[k] = score
        print(f"  k={k:>2}  silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\n  Best k = {best_k}  (silhouette={scores[best_k]:.4f})")

    # Fit final model
    print(f"\n[2/3] Fitting final KMeans (k={best_k})")
    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = km.fit_predict(X)

    # Top terms per cluster
    feature_names = vectorizer.get_feature_names_out()
    print("\n  Top 10 terms per cluster:")
    cluster_terms = {}
    for c in range(best_k):
        center = km.cluster_centers_[c]
        top_idx = np.argsort(center)[-10:][::-1]
        terms = [feature_names[i] for i in top_idx]
        cluster_terms[c] = terms
        size = int((cluster_labels == c).sum())
        print(f"  Cluster {c} ({size:>5} docs): {', '.join(terms)}")

    # Save artifacts
    joblib.dump({"vectorizer": vectorizer, "model": km},
                MODELS_DIR / "cluster_kmeans.joblib")
    out = df[["job_id", "title", "seniority_class"]].copy()
    out["cluster"] = cluster_labels
    out.to_csv(REPORTS_DIR / "cluster_assignments.csv", index=False)

    # Silhouette plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(scores.keys()), list(scores.values()), marker="o")
    ax.axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")
    ax.set_xlabel("k"); ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette Score vs Number of Clusters")
    ax.legend()
    fig.tight_layout(); fig.savefig(VIZ_DIR / "cluster_silhouette.png")
    plt.close(fig)

    # Cluster size bar
    sizes = pd.Series(cluster_labels).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=sizes.index, y=sizes.values, ax=ax)
    ax.set_xlabel("Cluster"); ax.set_ylabel("Postings")
    ax.set_title(f"Cluster Sizes (k={best_k})")
    fig.tight_layout(); fig.savefig(VIZ_DIR / "cluster_sizes.png")
    plt.close(fig)

    _save_json({
        "n_docs": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "silhouette_scores": scores,
        "best_k": int(best_k),
        "best_silhouette": scores[best_k],
        "cluster_terms": {str(k): v for k, v in cluster_terms.items()},
        "cluster_sizes": sizes.to_dict(),
    }, "cluster_metrics.json")
    print("[3/3] Saved clustering artifacts.")


# ------------------------------------------------------------------
# 5. CLI entrypoint
# ------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Job Market Analyzer — ML pipeline")
    parser.add_argument(
        "--task",
        choices=["salary", "seniority", "cluster", "all"],
        default="all",
        help="Which model to train (default: all)",
    )
    args = parser.parse_args()

    df = load_ml_data()

    if args.task in ("salary", "all"):
        run_salary_regression(df)
    if args.task in ("seniority", "all"):
        run_seniority_classification(df)
    if args.task in ("cluster", "all"):
        run_clustering(df)

    print("\n✔ Done.")


if __name__ == "__main__":
    main()