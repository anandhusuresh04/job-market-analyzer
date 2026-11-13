"""
nlp_cleaner.py
---------------
Loads the raw LinkedIn job postings dataset, drops irrelevant columns,
and cleans the job description text (HTML removal, normalization).

Target dataset: xanderios/linkedin-job-postings (Hugging Face)
"""

import re
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

# ------------------------------------------------------------------
# 1. Configuration (FIXED: Absolute Paths)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Keep only the most important columns for the ML pipeline
KEEP_COLUMNS = [
    "job_id",
    "title",
    "description",
    "formatted_experience_level",
    "pay_period",          
    "normalized_salary",   
    "max_salary",
    "min_salary",
    "med_salary",
]

COLUMN_ALIASES = {
    "description": ["description", "job_description"],
    "title": ["title", "job_title"],
    "formatted_experience_level": ["formatted_experience_level", "seniority_level"],
}


# ------------------------------------------------------------------
# 2. Loading
# ------------------------------------------------------------------
def load_raw_data(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load the first .csv or .parquet file found in the raw directory."""
    files = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No CSV or Parquet file found in {raw_dir}.")
    
    file_path = files[0]
    print(f"Loading data from: {file_path}")

    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        df = pd.read_parquet(file_path)

    print(f"Raw shape: {df.shape}")
    return df


# ------------------------------------------------------------------
# 3. Column handling
# ------------------------------------------------------------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to standard names and keep only relevant ones."""
    for standard_name, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                df = df.rename(columns={alias: standard_name})
                break

    existing_keep = [col for col in KEEP_COLUMNS if col in df.columns]
    dropped = set(df.columns) - set(existing_keep)
    if dropped:
        print(f"Dropping {len(dropped)} irrelevant columns.")

    return df[existing_keep].copy()


# ------------------------------------------------------------------
# 4. Text cleaning (FIXED: Regex preserving C++ and C#)
# ------------------------------------------------------------------
def strip_html(text: str) -> str:
    """Remove HTML tags and unescape HTML entities."""
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator=" ")

def clean_text(text: str) -> str:
    """Normalize whitespace, lowercase, and remove basic noise."""
    text = strip_html(text)
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # FIX: Added \+ and \# to preserve C++, C#, F#
    text = re.sub(r"[^a-z0-9\s\.,!?;:'\"()\-\+\#/&]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clean_descriptions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning to the 'description' column."""
    if "description" not in df.columns:
        raise KeyError("Column 'description' not found after standardization.")

    print("Cleaning job descriptions (HTML removal + text normalization)...")
    df["description_clean"] = df["description"].apply(clean_text)

    # Drop rows where the cleaned description is empty
    before = len(df)
    df = df[df["description_clean"].str.len() > 0].copy()
    print(f"Dropped {before - len(df)} rows with empty descriptions.")
    return df


# ------------------------------------------------------------------
# 5. Main pipeline
# ------------------------------------------------------------------
def main():
    df = load_raw_data()
    df = standardize_columns(df)
    df = clean_descriptions(df)

    output_path = PROCESSED_DIR / "job_postings_cleaned.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to: {output_path}")
    print(f"Final shape: {df.shape}")
    print("Columns:", list(df.columns))

if __name__ == "__main__":
    main()