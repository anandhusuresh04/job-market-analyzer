"""
visualization.py
-----------------
Polished, portfolio-grade charts for the Job Market Analyzer.

Regenerates every chart in visualizations/ from saved artifacts:
    - data/processed/job_postings_ml_ready.csv   (EDA)
    - reports/salary_metrics.json                (regression)
    - reports/salary_predictions.csv             (scatter)
    - reports/seniority_metrics.json             (classification)
    - reports/cluster_metrics.json               (clustering)

No ML re-training: one quick LR refit reproduces the confusion matrix
with the identical split (same random_state, same stratify), so numbers
match reports/seniority_metrics.json exactly.

Run:    python src/visualization.py
Output: visualizations/*.png (overwritten)

Style choices:
    - Per-category color palette, consistent across all charts
    - Count + percentage labels on bars
    - Subtitle with N (sample size) for context
    - Clean spines; horizontal bars for long labels
    - Dollar-formatted axes on salary charts
    - 2D density for scatter instead of overplotted dots
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 0. Config — keep in sync with ml_models.py
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR  = PROJECT_ROOT / "reports"
VIZ_DIR      = PROJECT_ROOT / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TFIDF_KW = dict(
    max_features=15_000, min_df=5, max_df=0.85,
    ngram_range=(1, 2), stop_words="english",
)

# Consistent palette — one color per semantic category
COLORS = {
    "Entry":      "#2E86AB",   # blue
    "Mid-Senior": "#A23B72",   # magenta
    "Director":   "#F18F01",   # orange
    "neutral":    "#5C6B73",   # slate
    "accent":     "#EE6C4D",   # coral
    "muted":      "#98C1D9",   # light blue
    "grid":       "#E8EDF0",
}
SENIORITY_ORDER = ["Entry", "Mid-Senior", "Director"]

SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "rust",
    "ruby", "php", "scala",
    "sql", "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow",
    "spark", "hadoop", "tableau", "power bi", "excel", "airflow", "dbt",
    "snowflake", "bigquery", "redshift",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "jenkins", "linux", "git",
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring boot",
    "machine learning", "deep learning", "nlp", "computer vision",
    "statistics", "a/b testing",
]

SKILL_CATEGORIES = {
    "Languages":    {"python", "java", "javascript", "typescript", "c++",
                     "c#", "rust", "ruby", "php", "scala"},
    "Data/ML":      {"sql", "pandas", "numpy", "scikit-learn", "pytorch",
                     "tensorflow", "spark", "hadoop", "tableau", "power bi",
                     "excel", "airflow", "dbt", "snowflake", "bigquery",
                     "redshift", "machine learning", "deep learning", "nlp",
                     "computer vision", "statistics", "a/b testing"},
    "Cloud/DevOps": {"aws", "azure", "gcp", "docker", "kubernetes",
                     "terraform", "jenkins", "linux", "git"},
    "Web":          {"react", "angular", "vue", "node.js", "django", "flask",
                     "fastapi", "spring boot"},
}
CATEGORY_COLORS = {
    "Languages":    "#2E86AB",
    "Data/ML":      "#A23B72",
    "Cloud/DevOps": "#F18F01",
    "Web":          "#5C6B73",
}


# ------------------------------------------------------------------
# 1. Shared helpers
# ------------------------------------------------------------------
def _setup_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#CCD6DC",
        "axes.linewidth": 0.8,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.titlepad": 16,
        "axes.labelsize": 11,
        "axes.labelcolor": "#333333",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "font.family": "sans-serif",
    })


def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(VIZ_DIR / name, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {name}")


def _strip_spines(ax, keep=("bottom", "left")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def _dollar_fmt(x, _pos):
    if x >= 1_000_000: return f"${x/1_000_000:.1f}M"
    if x >= 1_000:     return f"${x/1_000:.0f}k"
    return f"${x:.0f}"


def _subtitle(ax, text: str) -> None:
    ax.text(0.0, 1.02, text, transform=ax.transAxes,
            fontsize=9.5, color="#666666", style="italic")


def _skill_pattern(skill: str) -> str:
    if skill in ("c++", "c#"):
        return re.escape(skill)
    return r"\b" + re.escape(skill) + r"\b"


# ------------------------------------------------------------------
# 2. EDA charts
# ------------------------------------------------------------------
def chart_seniority_distribution(df: pd.DataFrame) -> None:
    counts = df["seniority_class"].value_counts()
    counts = counts.reindex([c for c in SENIORITY_ORDER if c in counts.index])
    total = int(counts.sum())
    labels = counts.index.tolist()
    values = counts.values
    pcts = values / total * 100

    fig, ax = plt.subplots(figsize=(9, 3.6))
    y = np.arange(len(labels))[::-1]
    colors = [COLORS[l] for l in labels]
    ax.barh(y, values, color=colors, height=0.62, edgecolor="white")
    for yi, v, p, l in zip(y, values, pcts, labels):
        ax.text(v + total * 0.012, yi, f"{v:,}  ({p:.1f}%)",
                va="center", fontsize=10.5, color="#222222",
                fontweight="semibold")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlim(0, max(values) * 1.30)
    ax.set_xlabel("Number of postings")
    ax.set_title("Job Postings by Seniority Class")
    _subtitle(ax, f"N = {total:,} labeled postings · collapsed from 7 raw LinkedIn seniority levels")
    ax.set_xticks([])
    _strip_spines(ax, keep=("left",))
    _save(fig, "seniority_distribution.png")


def chart_salary_distribution(reg_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(reg_df["annual_salary_est"], bins=55, kde=True, ax=ax,
                 color=COLORS["Entry"], edgecolor="white", alpha=0.85)
    median = reg_df["annual_salary_est"].median()
    mean   = reg_df["annual_salary_est"].mean()
    ax.axvline(median, color=COLORS["accent"], linestyle="--", lw=1.5,
               label=f"Median = {_dollar_fmt(median, None)}")
    ax.axvline(mean,   color=COLORS["Director"], linestyle=":", lw=1.5,
               label=f"Mean   = {_dollar_fmt(mean, None)}")
    ax.xaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    ax.set_xlabel("Annual salary (USD)")
    ax.set_ylabel("Number of postings")
    ax.set_title("Distribution of Annualized Salaries")
    _subtitle(ax, f"N = {len(reg_df):,} postings · filtered to $15k–$500k · log-transform used for modeling")
    ax.legend(frameon=False, loc="upper right")
    _strip_spines(ax); ax.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax.set_axisbelow(True)
    _save(fig, "salary_distribution.png")


def chart_salary_by_seniority(reg_df: pd.DataFrame) -> None:
    df = reg_df.dropna(subset=["seniority_class"]).copy()
    order = [c for c in SENIORITY_ORDER if c in df["seniority_class"].unique()]
    palette = {c: COLORS[c] for c in order}

    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.violinplot(data=df, x="seniority_class", y="annual_salary_est",
                   order=order, hue="seniority_class", palette=palette,
                   inner=None, cut=0, linewidth=0.8, saturation=0.75,
                   legend=False, ax=ax)
    sns.stripplot(data=df, x="seniority_class", y="annual_salary_est",
                  order=order, color="#222222", size=1.6, alpha=0.18,
                  jitter=0.28, ax=ax)

    medians = df.groupby("seniority_class")["annual_salary_est"].median().reindex(order)
    for i, (cls, m) in enumerate(medians.items()):
        ax.text(i, m, f"  {_dollar_fmt(m, None)}", va="center", fontsize=10.5,
                fontweight="bold", color="white",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=COLORS[cls],
                          edgecolor="none"))

    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    ax.set_xlabel(""); ax.set_ylabel("Annual salary (USD)")
    ax.set_title("Annual Salary by Seniority Class")
    _subtitle(ax, "Violin = distribution · dots = individual postings · labels = median")
    _strip_spines(ax); ax.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax.set_axisbelow(True)
    _save(fig, "salary_by_seniority.png")


def chart_top_skills(df: pd.DataFrame) -> None:
    series = df["description_clean"]
    counts = {s: int(series.str.contains(_skill_pattern(s), regex=True).sum())
              for s in SKILL_KEYWORDS}
    ser = pd.Series(counts).sort_values(ascending=False).head(15)
    total = len(df)

    def _cat(skill):
        for cat, members in SKILL_CATEGORIES.items():
            if skill in members:
                return cat
        return "Languages"

    labels = ser.index.tolist()
    values = ser.values
    cats   = [_cat(l) for l in labels]
    colors = [CATEGORY_COLORS[c] for c in cats]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colors, height=0.7, edgecolor="white")
    for yi, v, l in zip(y, values, labels):
        pct = v / total * 100
        ax.text(v + total * 0.004, yi, f"{v:,}  ({pct:.1f}%)",
                va="center", fontsize=10, color="#222222")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Number of postings mentioning skill")
    ax.set_title("Top 15 Skills Mentioned in Job Descriptions")
    _subtitle(ax, f"N = {total:,} postings · footnote: 'excel' includes verb usages (inflated ~20–30%)")
    ax.set_xticks([])
    handles = [plt.Rectangle((0,0),1,1,color=c) for c in CATEGORY_COLORS.values()]
    ax.legend(handles, CATEGORY_COLORS.keys(), frameon=False,
              loc="lower right", fontsize=9, title="Category", title_fontsize=9)
    _strip_spines(ax, keep=("left",))
    _save(fig, "top_skills.png")


def chart_pay_period(df: pd.DataFrame) -> None:
    counts = df["pay_period"].fillna("MISSING").value_counts()
    total = int(counts.sum())
    fig, ax = plt.subplots(figsize=(9, 3.4))
    colors = [COLORS["muted"] if k == "MISSING" else COLORS["Entry"]
              for k in counts.index]
    x = np.arange(len(counts))
    ax.bar(x, counts.values, color=colors, edgecolor="white")
    for xi, v in zip(x, counts.values):
        ax.text(xi, v, f"{v:,}\n({v/total*100:.1f}%)", ha="center", va="bottom",
                fontsize=9.5, color="#222222")
    ax.set_xticks(x); ax.set_xticklabels(counts.index, fontsize=10)
    ax.set_ylim(0, max(counts.values) * 1.18)
    ax.set_ylabel("Postings")
    ax.set_title("Pay Period Coverage")
    _subtitle(ax, "60% of postings lack salary data — regression trains on the 33% with valid annual figures")
    ax.set_yticks([])
    _strip_spines(ax, keep=("bottom",))
    _save(fig, "pay_period_distribution.png")


def chart_description_length(df: pd.DataFrame) -> None:
    d = df.dropna(subset=["seniority_class"]).copy()
    d["desc_len"] = d["description_clean"].str.len()
    order = [c for c in SENIORITY_ORDER if c in d["seniority_class"].unique()]
    palette = {c: COLORS[c] for c in order}

    fig, ax = plt.subplots(figsize=(9, 4))
    sns.boxplot(data=d, x="seniority_class", y="desc_len", order=order,
                hue="seniority_class", palette=palette, width=0.5,
                fliersize=2, legend=False, ax=ax)
    for i, cls in enumerate(order):
        med = d.loc[d["seniority_class"] == cls, "desc_len"].median()
        ax.text(i, med, f"{med:,.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color="#222222")
    ax.set_xlabel(""); ax.set_ylabel("Description length (characters)")
    ax.set_title("Job Description Length by Seniority Class")
    _subtitle(ax, "Median annotated · longer descriptions tend to indicate more senior or specialist roles")
    _strip_spines(ax); ax.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax.set_axisbelow(True)
    _save(fig, "description_length.png")


# ------------------------------------------------------------------
# 3. Regression chart
# ------------------------------------------------------------------
def chart_salary_pred_vs_actual() -> None:
    preds = pd.read_csv(REPORTS_DIR / "salary_predictions.csv")
    salary = json.loads((REPORTS_DIR / "salary_metrics.json").read_text())
    r2   = salary["ridge"]["r2"]
    rmse = salary["ridge"]["rmse"]

    fig, ax = plt.subplots(figsize=(7, 6.4))
    # 2D hexbin shows density without overplotting 2,201 points
    hb = ax.hexbin(preds["actual"], preds["ridge_pred"],
                   gridsize=38, cmap="Blues", mincnt=1, linewidths=0.2)
    lims = [preds[["actual", "ridge_pred"]].min().min() * 0.95,
            preds[["actual", "ridge_pred"]].max().max() * 1.02]
    ax.plot(lims, lims, "--", color=COLORS["accent"], lw=1.4,
            label="Perfect prediction")
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.xaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    ax.yaxis.set_major_formatter(FuncFormatter(_dollar_fmt))
    ax.set_xlabel("Actual annual salary")
    ax.set_ylabel("Predicted annual salary")
    ax.set_title("Salary Prediction — RidgeCV")
    _subtitle(ax, f"Test set N = {len(preds):,} · darker = more postings · "
                  f"R² = {r2:.3f},  RMSE = {_dollar_fmt(rmse, None)}")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    _strip_spines(ax)
    cb = fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label("Count", fontsize=9)
    _save(fig, "salary_pred_vs_actual.png")


# ------------------------------------------------------------------
# 4. Classification chart — re-derives LR split for confusion matrix
# ------------------------------------------------------------------
def chart_seniority_confusion(df: pd.DataFrame) -> None:
    print("  (refitting quick LR for confusion matrix — identical split)")
    sub = df.dropna(subset=["seniority_class"]).copy()
    X_text = sub["description_clean"].values
    y = sub["seniority_class"].values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    vec = TfidfVectorizer(**TFIDF_KW)
    Xtr = vec.fit_transform(X_tr); Xte = vec.transform(X_te)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(Xtr, y_tr)
    y_pred = clf.predict(Xte)

    labels = [c for c in SENIORITY_ORDER if c in np.unique(y)]
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels,
                cbar_kws={"label": "Row-normalized proportion"},
                linewidths=1.2, linecolor="white",
                annot_kws={"fontsize": 12, "fontweight": "bold"},
                ax=ax)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_title("Seniority Confusion Matrix — LogisticRegression")
    _subtitle(ax, "Cells = raw counts · color = row-normalized proportion (diagonal = recall)")
    ax.set_xticklabels(labels, rotation=0, fontsize=10.5)
    ax.set_yticklabels(labels, rotation=0, fontsize=10.5)
    _save(fig, "seniority_confusion_matrix.png")


# ------------------------------------------------------------------
# 5. Clustering charts
# ------------------------------------------------------------------
def chart_cluster_silhouette() -> None:
    clu = json.loads((REPORTS_DIR / "cluster_metrics.json").read_text())
    ks = sorted(int(k) for k in clu["silhouette_scores"].keys())
    ss = [clu["silhouette_scores"][str(k)] for k in ks]
    best_k = clu["best_k"]

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    ax.plot(ks, ss, "-o", color=COLORS["Entry"], lw=2, markersize=9,
            markerfacecolor="white", markeredgewidth=2)
    for k, s in zip(ks, ss):
        ax.annotate(f"{s:.4f}", (k, s), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, color="#444444")
    ax.axvline(best_k, color=COLORS["accent"], linestyle="--", lw=1.3,
               label=f"best k = {best_k}")
    ax.set_xticks(ks)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.set_title("Silhouette Score vs. Number of Clusters")
    _subtitle(ax, "Low absolute scores (0.01–0.02) are normal for sparse text — "
                  "cluster quality assessed semantically")
    ax.legend(frameon=False)
    _strip_spines(ax); ax.grid(axis="y", color=COLORS["grid"], lw=0.7)
    ax.set_axisbelow(True)
    _save(fig, "cluster_silhouette.png")


def chart_cluster_sizes() -> None:
    clu = json.loads((REPORTS_DIR / "cluster_metrics.json").read_text())
    sizes = {int(k): int(v) for k, v in clu["cluster_sizes"].items()}
    terms = clu["cluster_terms"]
    best_terms = {int(k): v for k, v in terms.items()}

    # Interpretable labels from top-3 terms
    INTERPRET = {
        0: "Travel nursing agency",
        1: "Manufacturing / industrial",
        2: "Office / insurance services",
        3: "Software / data engineering",
        4: "Marketing / product",
        5: "Hospital clinical / nursing",
        6: "Sales / business dev",
        7: "Accounting / finance",
        8: "Retail store ops",
        9: "Construction / project mgmt",
    }
    labels = [INTERPRET.get(k, f"Cluster {k}") for k in sizes.keys()]
    values = list(sizes.values())
    order = np.argsort(values)[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(labels))[::-1]
    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(labels)))
    ax.barh(y, values, color=colors, height=0.72, edgecolor="white")
    total = sum(values)
    for yi, v in zip(y, values):
        ax.text(v + total * 0.006, yi, f"{v:,}  ({v/total*100:.1f}%)",
                va="center", fontsize=10, color="#222222")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Number of postings")
    ax.set_title("Labor Market Segments (KMeans, k=10)")
    _subtitle(ax, "Interpretable labels derived from each cluster's top TF-IDF terms")
    ax.set_xticks([])
    _strip_spines(ax, keep=("left",))
    _save(fig, "cluster_sizes.png")


# ------------------------------------------------------------------
# 6. Main
# ------------------------------------------------------------------
def main() -> None:
    _setup_theme()
    print("Loading data…")
    df = pd.read_csv(DATA_DIR / "job_postings_ml_ready.csv")
    df = df[df["description_clean"].str.len() >= 50].copy()
    reg_df = df[df["is_salary_valid"]].copy()
    print(f"  Loaded {len(df):,} postings ({len(reg_df):,} with valid salary)")

    print("\nGenerating charts…")
    chart_seniority_distribution(df)
    chart_salary_distribution(reg_df)
    chart_salary_by_seniority(reg_df)
    chart_top_skills(df)
    chart_pay_period(df)
    chart_description_length(df)
    chart_salary_pred_vs_actual()
    chart_seniority_confusion(df)
    chart_cluster_silhouette()
    chart_cluster_sizes()

    print(f"\n✔ All 10 charts saved to {VIZ_DIR}")
    print("  Re-run the notebook to display the updated PNGs.")


if __name__ == "__main__":
    main()