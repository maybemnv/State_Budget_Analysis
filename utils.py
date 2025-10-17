
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def sanitize_filename(description: str, table_num: str, page_num: int, max_length: int = 100) -> str:
    """
    Create a filesystem-safe filename with strict sanitization.
    Removes all problematic characters including : / \ ? * < > | "
    """
    # Remove all non-alphanumeric characters except spaces, hyphens, and underscores
    safe_description = re.sub(r'[^\w\s\-]', '', description)
    # Replace multiple spaces with single underscore
    safe_description = re.sub(r'\s+', '_', safe_description.strip())
    # Remove leading/trailing underscores
    safe_description = safe_description.strip('_')
    
    # Create filename
    safe_table_num = table_num.replace('.', '_')
    filename = f"Table_{safe_table_num}_page_{page_num}_{safe_description}"
    
    # Truncate to max length
    if len(filename) > max_length:
        # Keep the table number and page, truncate description
        prefix = f"Table_{safe_table_num}_page_{page_num}_"
        remaining = max_length - len(prefix)
        if remaining > 10:  # Ensure we have some description left
            filename = prefix + safe_description[:remaining]
        else:
            filename = f"Table_{safe_table_num}_page_{page_num}"
    
    # Ensure filename is not empty
    if not filename:
        filename = f"Table_{safe_table_num}_page_{page_num}"
    
    return filename

def get_output_directory(pdf_path: str, add_timestamp: bool = False) -> Path:
    """
    Create output directory with optional timestamp to prevent overwrites.
    """
    pdf_name = Path(pdf_path).stem
    base_output_dir = Path('extracted_tables')
    
    if add_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_output_dir / f"{pdf_name}_{timestamp}"
    else:
        output_dir = base_output_dir / pdf_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def print_extraction_summary(titled_tables: List[Dict[str, Any]]) -> None:
    """
    Print a formatted summary of extracted tables.
    """
    if not titled_tables:
        return
    
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    for table in titled_tables:
        print(f"\n📊 Table {table['table_number']}: {table['description']}")
        print(f"   📄 Page: {table['page']}")
        print(f"   📁 File: {table['filename']}.xlsx")
        print(f"   📐 Size: {table['shape'][0]} rows × {table['shape'][1]} columns")
    
    print("\n" + "=" * 70)
