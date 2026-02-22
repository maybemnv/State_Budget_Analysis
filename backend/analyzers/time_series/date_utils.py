import re
import pandas as pd


def parse_financial_year(value: str) -> int | None:
    """Return the starting calendar year for a financial year string.

    Handles formats: '2023', 'FY23', 'FY2023', '2023-24', '2023/24'.
    Returns None when the value cannot be parsed.
    """
    s = str(value).strip().upper()

    # Plain 4-digit year
    if re.fullmatch(r"\d{4}", s):
        return int(s)

    # FY23 or FY2023
    m = re.fullmatch(r"FY(\d{2}|\d{4})", s)
    if m:
        year = int(m.group(1))
        return year if year > 100 else 2000 + year

    # 2023-24 or 2023/24
    m = re.fullmatch(r"(\d{4})[-/]\d{2,4}", s)
    if m:
        return int(m.group(1))

    return None


def extract_year_columns(df: pd.DataFrame) -> list[str]:
    """Return column names from *df* that represent year or financial-year periods."""
    year_cols = []
    for col in df.columns:
        if parse_financial_year(str(col)) is not None:
            year_cols.append(col)
    return year_cols
