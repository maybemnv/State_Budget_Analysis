import re
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any, Union
import pandas as pd

# Common date formats to try when parsing dates
COMMON_DATE_FORMATS = [
    '%Y-%m-%d', '%d-%m-%Y', '%m/%d/%Y', '%Y/%m/%d',
    '%Y%m%d', '%d%m%Y', '%m%d%Y',
    '%Y-%m', '%Y',
    '%b %Y', '%B %Y',
    '%d-%b-%Y', '%d-%B-%Y',
    '%b-%y', '%B-%y',
    '%Y-Q%q',  # Quarterly formats
    'FY%y', 'FY%Y',  # Fiscal year formats
]

# Financial year patterns (e.g., 2021-22, 2021/22, FY2021-22, etc.)
FINANCIAL_YEAR_PATTERNS = [
    (r'(\d{4})[-/](\d{2})$', lambda m: f"20{m.group(1)}-07-01"),  # 2021-22
    (r'FY(\d{2})[-/](\d{2})', lambda m: f"20{m.group(1)}-04-01"),  # FY21-22
    (r'(\d{4})-\d{2}$', lambda m: f"{m.group(1)}-04-01"),  # 2021-22 (full year)
    (r'FY(\d{4})', lambda m: f"{m.group(1)}-04-01"),  # FY2021
]

def parse_financial_year(year_str: str) -> Optional[datetime]:
    """
    Parse financial year strings into datetime objects.
    
    Args:
        year_str: String representing a financial year (e.g., '2021-22', 'FY21-22')
        
    Returns:
        datetime object if parsing is successful, None otherwise
    """
    if not isinstance(year_str, str):
        year_str = str(year_str)
        
    year_str = year_str.strip().upper()
    
    # Try standard date formats first
    for fmt in COMMON_DATE_FORMATS:
        try:
            return datetime.strptime(year_str, fmt)
        except ValueError:
            continue
    
    # Try financial year patterns
    for pattern, date_func in FINANCIAL_YEAR_PATTERNS:
        match = re.match(pattern, year_str, re.IGNORECASE)
        if match:
            try:
                return pd.to_datetime(date_func(match))
            except:
                continue
    
    return None

def detect_date_format(series: pd.Series) -> Tuple[Optional[str], float]:
    """
    Detect the most likely date format for a pandas Series.
    
    Args:
        series: Pandas Series containing date strings
        
    Returns:
        Tuple of (best_format, confidence) where confidence is between 0 and 1
    """
    if not pd.api.types.is_string_dtype(series):
        return None, 0.0
        
    sample = series.dropna().head(50)  # Check up to 50 non-null values
    if len(sample) == 0:
        return None, 0.0
    
    format_scores = {}
    
    for fmt in COMMON_DATE_FORMATS:
        try:
            parsed = pd.to_datetime(sample, format=fmt, errors='coerce')
            valid_count = parsed.notna().sum()
            if valid_count > 0:
                format_scores[fmt] = valid_count / len(sample)
        except:
            continue
    
    # Check financial year patterns if no good matches found
    if not format_scores or max(format_scores.values()) < 0.5:
        fy_count = 0
        for val in sample:
            if parse_financial_year(val) is not None:
                fy_count += 1
        
        if fy_count / len(sample) > 0.5:  # If more than 50% match financial year patterns
            return 'financial_year', fy_count / len(sample)
    
    # Return the best matching format
    if format_scores:
        best_format = max(format_scores.items(), key=lambda x: x[1])
        return best_format
    
    return None, 0.0

def extract_year_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify columns that contain year values.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names that appear to contain year values
    """
    year_cols = []
    
    for col in df.columns:
        col_str = str(col).strip()
        
        # Check if column name is a year
        if col_str.isdigit() and 1900 <= int(col_str) <= 2100:
            year_cols.append(col)
        # Check for financial year in column name
        elif any(fy in col_str.upper() for fy in ['FY', 'FINANCIAL_YEAR', 'FISCAL_YEAR']):
            year_cols.append(col)
    
    return year_cols
