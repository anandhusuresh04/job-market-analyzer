"""
analysis.py
------------
Exploratory Data Analysis for the Job Market Analyzer.

Responsibilities
----------------
1. Derive regression target : annual_salary_est  (annualized USD midpoint)
2. Derive 3-class label     : seniority_class   (Entry / Mid-Senior / Director)
3. Flag salary outliers     : is_salary_valid   (True if in [$15k, $500k])
4. Print EDA report to console
5. Save EDA charts to visualizations/
6. Persist ML-ready dataset to data/processed/job_postings_ml_ready.csv

The CSV saves *all* rows with an is_salary_valid flag — downstream ML
code filters on that flag rather than relying on rows having been
silently dropped. This prevents the class of bug where analysis-time
filters do not propagate to model-training time.

Inputs : data/processed/job_postings_cleaned.csv
Outputs: data/processed/job_postings_ml_ready.csv
         visualizations/*.png
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless-safe backend for script runs
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
VIZ_DIR = PROJECT_ROOT / "visualizations"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "job_postings_cleaned.csv"
OUTPUT_PATH = PROCESSED_DIR / "job_postings_ml_ready.csv"

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110

# ------------------------------------------------------------------
# 1. Salary annualization
# ------------------------------------------------------------------
PAY_PERIOD_MULTIPLIER = {
    "YEARLY": 1,
    "MONTHLY": 12,
    "WEEKLY": 52,
    "HOURLY": 2080,        # 40 h/week * 52 weeks
    # "ONCE" intentionally excluded: one-off payment, not an annual salary
}
SALARY_MIN = 15_000
SALARY_MAX = 500_000


def annualize_salary(df: pd.DataFrame) -> pd.DataFrame:
    """Annualize the (min+max)/2 midpoint and flag plausible salaries.

    The flag `is_salary_valid` is the single source of truth for which
    rows are eligible for salary regression. It is stored in the CSV so
    downstream code (ml_models.py) never has to re-derive the filter.
    """
    df = df.copy()
    mult = df["pay_period"].map(PAY_PERIOD_MULTIPLIER)
    midpoint = (df["min_salary"] + df["max_salary"]) / 2.0
    df["annual_salary_est"] = midpoint * mult

    df["is_salary_valid"] = (
        df["annual_salary_est"].between(SALARY_MIN, SALARY_MAX)
    )

    n_computed = df["annual_salary_est"].notna().sum()
    n_valid = int(df["is_salary_valid"].sum())
    print(f"[salary] annual_salary_est computed for {n_computed:,} rows "
          f"({n_computed / len(df):.1%} of dataset).")
    print(f"[salary] is_salary_valid=True for {n_valid:,} rows "
          f"({n_valid / len(df):.1%}) after [${SALARY_MIN:,}, "
          f"${SALARY_MAX:,}] filter.")
    return df


def regression_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for salary regression, filtered via is_salary_valid."""
    sub = df[df["is_salary_valid"]].copy()
    print(f"[salary] regression-ready rows after sanity filter: {len(sub):,}")
    return sub


# ------------------------------------------------------------------
# 2. Seniority -> 3 classes
# ------------------------------------------------------------------
SENIORITY_MAP = {
    "Internship": "Entry",
    "Entry level": "Entry",
    "Associate": "Entry",
    "Mid-Senior level": "Mid-Senior",
    "Director": "Director",
    "Executive": "Director",
}


def map_seniority(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["seniority_class"] = df["formatted_experience_level"].map(SENIORITY_MAP)
    n_nan = df["seniority_class"].isna().sum()
    print(f"[seniority] mapped to 3 classes; {n_nan:,} rows unlabeled "
          f"(kept in file, excluded from classification set).")
    return df


# ------------------------------------------------------------------
# 3. Skill frequency (EDA only; TF-IDF handles ML features)
# ------------------------------------------------------------------
SKILL_KEYWORDS = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "rust",
    "ruby", "php", "scala",
    # Data / ML
    "sql", "pandas", "numpy", "scikit-learn", "pytorch", "tensorflow", "spark",
    "hadoop", "tableau", "power bi", "excel", "airflow", "dbt", "snowflake",
    "bigquery", "redshift",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins",
    "linux", "git",
    # Web / backend
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring boot",
    # Concepts
    "machine learning", "deep learning", "nlp", "computer vision",
    "statistics", "a/b testing",
]


def _skill_pattern(skill: str) -> str:
    """Literal match for c++/c#; word-boundary match for everything else."""
    if skill in ("c++", "c#"):
        return re.escape(skill)
    return r"\b" + re.escape(skill) + r"\b"


def skill_frequency(series: pd.Series) -> pd.Series:
    """Count postings mentioning each curated skill, with live progress."""
    counts = {}
    total = len(SKILL_KEYWORDS)
    for i, s in enumerate(SKILL_KEYWORDS, 1):
        counts[s] = int(series.str.contains(_skill_pattern(s), regex=True).sum())
        print(f"  [{i:>2}/{total}] {s:<18} {counts[s]:>6,}", flush=True)
    return pd.Series(counts).sort_values(ascending=False)


# ------------------------------------------------------------------
# 4. Charts
# ------------------------------------------------------------------
def _save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(VIZ_DIR / name)
    plt.close(fig)


def save_seniority_distribution(df: pd.DataFrame) -> None:
    counts = df["seniority_class"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Postings by Seniority Class")
    ax.set_ylabel("Count"); ax.set_xlabel("")
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    _save(fig, "seniority_distribution.png")


def save_salary_distribution(reg_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(reg_df["annual_salary_est"], bins=60, kde=True, ax=ax)
    ax.set_title("Annualized Salary Distribution (regression subset)")
    ax.set_xlabel("Annual salary (USD)")
    _save(fig, "salary_distribution.png")

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=reg_df, x="seniority_class", y="annual_salary_est", ax=ax)
    ax.set_title("Annual Salary by Seniority Class")
    ax.set_ylabel("Annual salary (USD)"); ax.set_xlabel("")
    _save(fig, "salary_by_seniority.png")


def save_pay_period(df: pd.DataFrame) -> None:
    counts = df["pay_period"].fillna("MISSING").value_counts()
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax)
    ax.set_title("Pay Period Distribution")
    ax.set_ylabel("Count"); ax.set_xlabel("")
    _save(fig, "pay_period_distribution.png")


def save_top_skills(skill_counts: pd.Series, top_n: int = 20) -> None:
    top = skill_counts.head(top_n)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax)
    ax.set_title(f"Top {top_n} Skills Mentioned in Descriptions")
    ax.set_xlabel("Number of postings"); ax.set_ylabel("")
    _save(fig, "top_skills.png")


def save_description_length(df: pd.DataFrame) -> None:
    df = df.copy()
    df["desc_len"] = df["description_clean"].str.len()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=df, x="seniority_class", y="desc_len", ax=ax)
    ax.set_title("Description Length by Seniority Class")
    ax.set_xlabel(""); ax.set_ylabel("Characters")
    _save(fig, "description_length.png")


# ------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------
def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} not found. Run src/nlp_cleaner.py first.")

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded cleaned data: {df.shape[0]:,} rows, {df.shape[1]} columns\n")

    df = annualize_salary(df)
    df = map_seniority(df)
    reg_df = regression_subset(df)

    print("\n--- Annual salary stats (regression subset) ---")
    print(reg_df["annual_salary_est"].describe().round(0).to_string())

    print("\n--- Salary by seniority class (regression subset only) ---")
    print(
        reg_df.dropna(subset=["seniority_class"])
              .groupby("seniority_class")["annual_salary_est"]
              .agg(["count", "median", "mean"]).round(0).to_string()
    )

    print("\n--- Seniority class distribution (classification set) ---")
    print(df["seniority_class"].value_counts(dropna=False).to_string())

    print("\n--- Skill frequencies (EDA) ---")
    skills = skill_frequency(df["description_clean"])
    print("\nTop 15 skills:")
    print(skills.head(15).to_string())

    print("\n--- Generating charts ---")
    for name, fn, arg in [
        ("seniority_distribution",  save_seniority_distribution, df),
        ("salary_distribution",     save_salary_distribution,    reg_df),
        ("pay_period_distribution", save_pay_period,             df),
        ("top_skills",              save_top_skills,             skills),
        ("description_length",      save_description_length,     df),
    ]:
        print(f"  -> {name}", flush=True)
        fn(arg)
    print(f"Saved 5 charts to {VIZ_DIR}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved ML-ready dataset -> {OUTPUT_PATH}")
    print(f"Final shape: {df.shape}")
    print("Columns:", list(df.columns))
    print(f"Rows with is_salary_valid=True: "
          f"{int(df['is_salary_valid'].sum()):,} "
          f"/ {len(df):,}")


if __name__ == "__main__":
    main()