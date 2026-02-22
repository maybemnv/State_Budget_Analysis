import re
from datetime import datetime
from typing import Optional, Tuple
import pandas as pd


COMMON_DATE_FORMATS = [
    "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%Y%m%d", "%Y-%m", "%Y",
    "%b %Y", "%B %Y", "%d-%b-%Y", "%d-%B-%Y",
]

FINANCIAL_YEAR_PATTERNS = [
    (r"(\d{4})[-/](\d{2})$", lambda m: f"{m.group(1)}-07-01"),
    (r"FY(\d{2})[-/](\d{2})", lambda m: f"20{m.group(1)}-04-01"),
    (r"FY(\d{4})", lambda m: f"{m.group(1)}-04-01"),
]


def parse_financial_year(year_str: str) -> Optional[datetime]:
    if not isinstance(year_str, str):
        year_str = str(year_str)
    year_str = year_str.strip().upper()
    for fmt in COMMON_DATE_FORMATS:
        try:
            return datetime.strptime(year_str, fmt)
        except ValueError:
            continue
    for pattern, date_func in FINANCIAL_YEAR_PATTERNS:
        match = re.match(pattern, year_str, re.IGNORECASE)
        if match:
            try:
                return pd.to_datetime(date_func(match))
            except Exception:
                continue
    return None


def detect_date_format(series: pd.Series) -> Tuple[Optional[str], float]:
    if not pd.api.types.is_string_dtype(series):
        return None, 0.0
    sample = series.dropna().head(50)
    if len(sample) == 0:
        return None, 0.0
    scores: dict[str, float] = {}
    for fmt in COMMON_DATE_FORMATS:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors="coerce")
            valid = parsed.notna().sum()
            if valid:
                scores[fmt] = valid / len(sample)
        except Exception:
            continue
    if not scores or max(scores.values()) < 0.5:
        fy_count = sum(1 for v in sample if parse_financial_year(str(v)) is not None)
        if fy_count / len(sample) > 0.5:
            return "financial_year", fy_count / len(sample)
    if scores:
        best = max(scores.items(), key=lambda x: x[1])
        return best
    return None, 0.0


def extract_year_columns(df: pd.DataFrame) -> list[str]:
    year_cols = []
    for col in df.columns:
        col_str = str(col).strip()
        if col_str.isdigit() and 1900 <= int(col_str) <= 2100:
            year_cols.append(col)
        elif any(fy in col_str.upper() for fy in ["FY", "FINANCIAL_YEAR", "FISCAL_YEAR"]):
            year_cols.append(col)
    return year_cols
