import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import pdfplumber

@dataclass
class TableExtractionSettings:
    vertical_strategy: str = "text"
    horizontal_strategy: str = "text"
    snap_tolerance: int = 3
    join_tolerance: int = 3
    edge_min_length: int = 3
    min_words_horizontal: int = 3

DEFAULT_SETTINGS = TableExtractionSettings()

def extract_table_data(page, settings: TableExtractionSettings = None) -> List[List[str]]:
    settings = settings or DEFAULT_SETTINGS
    try:
        return page.extract_tables(vars(settings))
    except Exception:
        return []

def clean_table_data(table_data: List[List[str]]) -> Optional[pd.DataFrame]:
    if not table_data or len(table_data) < 2:
        return None
    
    df = pd.DataFrame(table_data[1:], columns=table_data[0])
    df = df.dropna(how='all').reset_index(drop=True)
    return df if not df.empty else None

def find_table_titles(page) -> List[Dict[str, Any]]:
    titles = []
    text = page.extract_text() or ""
    for match in re.finditer(r'Table\s+([\d.]+):?\s*(.*?)(?=\n\n|$)', text, re.IGNORECASE | re.DOTALL):
        titles.append({
            'number': match.group(1).strip(),
            'description': match.group(2).strip(),
            'bbox': (0, 0, page.width, 100),
            'page': page.page_number
        })
    return titles