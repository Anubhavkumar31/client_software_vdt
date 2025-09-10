# utils/data_processing.py
import pandas as pd
from config.constants import NUMERIC_COLUMNS, COLUMN_VARIANTS, ROUNDING_PRECISION

def process_table_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize a DataFrame for use in defect/pipe tally tables.
    - Rounds numeric columns
    - Renames variant column headers
    - Drops duplicate unnamed columns
    """
    df = df.copy()

    # Standardize column names
    df.rename(columns={k: v for k, v in COLUMN_VARIANTS.items() if k in df.columns}, inplace=True)

    # Round numeric columns safely
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(ROUNDING_PRECISION)

    # Drop unnamed/empty columns
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df
