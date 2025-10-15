import pandas as pd
from typing import Iterable, Tuple


def validate_dataframe(df: pd.DataFrame) -> Tuple[bool, str]:
    """Basic checks for a DataFrame used in analysis flows."""
    if df is None:
        return False, "DataFrame is None"
    if not isinstance(df, pd.DataFrame):
        return False, "Input is not a pandas DataFrame"
    if df.empty:
        return False, "DataFrame is empty"
    return True, "ok"


def validate_columns(df: pd.DataFrame, required: Iterable[str]) -> Tuple[bool, str]:
    """Ensure all required columns exist in the DataFrame."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {', '.join(map(str, missing))}"
    return True, "ok"


