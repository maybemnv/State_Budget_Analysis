import pdfplumber
import pandas as pd
import re
from typing import List, Dict, Any, Optional, Tuple

TABLE_EXTRACTION_SETTINGS = [
    {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
    },
    {
        "vertical_strategy": "text",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
    },
    {
        "vertical_strategy": "lines",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
    },
    {
        "vertical_strategy": "text",
        "horizontal_strategy": "text",
        "snap_tolerance": 3,
        "join_tolerance": 3,
    },
]

def find_table_titles(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    """
    Find all table titles on a page.
    """
    titles = []
    pattern = r'Table\s+([0-9A-Za-z\.\-]+)\s*[:\-\s]\s*(.+?)(?=\n\n|\Z)'
    for match in re.finditer(pattern, page.extract_text(), re.IGNORECASE | re.MULTILINE):
        titles.append({
            'number': match.group(1),
            'description': match.group(2).strip(),
            'page_number': page.page_number,
            'bbox': (match.start(), page.height - match.end(), match.end(), page.height - match.start())
        })
    return titles

def extract_tables_from_page(page: pdfplumber.page.Page) -> List[Any]:
    """
    Extract all tables from a page using different strategies.
    """
    all_tables = []
    for settings in TABLE_EXTRACTION_SETTINGS:
        try:
            tables = page.find_tables(settings)
            if tables:
                all_tables.extend(tables)
        except Exception as e:
            print(f"Error extracting tables with settings {settings}: {e}")
    return all_tables

def match_tables_to_titles(tables: List[Any], titles: List[Dict[str, Any]], page_height: float) -> List[Dict[str, Any]]:
    """
    Match extracted tables to their titles based on proximity.
    """
    matched_tables = []
    for i, table in enumerate(tables):
        best_title = None
        min_distance = float('inf')
        for title in titles:
            distance = table.bbox[1] - title['bbox'][3]
            if 0 < distance < min_distance:
                min_distance = distance
                best_title = title
        
        if best_title:
            matched_tables.append({
                'title': best_title,
                'table': table,
            })
    return matched_tables

def clean_and_validate_table(table: List[List[str]]) -> Optional[pd.DataFrame]:
    """
    Clean and validate an extracted table.
    """
    if not table:
        return None
    
    df = pd.DataFrame(table)
    
    # Find header row
    header_row = 0
    for i, row in df.iterrows():
        # A header row is likely to have more unique values than data rows
        if row.nunique() > df.shape[1] / 2:
            header_row = i
            break
            
    df.columns = df.iloc[header_row]
    df = df.iloc[header_row + 1:]
    
    # Remove empty rows and columns
    df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df