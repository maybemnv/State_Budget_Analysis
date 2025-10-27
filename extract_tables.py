import re
import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd
import pdfplumber
from table_extractor import TableExtractionSettings, extract_table_data, clean_table_data, find_table_titles

class PDFTableExtractor:
    def __init__(self, pdf_path: str, output_dir: str = "extracted_tables"):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir) / self.pdf_path.stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logging()
        self.tables: List[Dict] = []
        self.settings = TableExtractionSettings(
            vertical_strategy="text",
            horizontal_strategy="text",
            snap_tolerance=3,
            join_tolerance=3,
            edge_min_length=3
        )

    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)

    def _process_page(self, page):
        try:
            # Find tables using the extractor
            tables = page.extract_tables({
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
                "snap_tolerance": 3,
                "join_tolerance": 3,
                "edge_min_length": 3,
            })

            # Find titles on the page
            titles = find_table_titles(page)
            
            for i, table in enumerate(tables, 1):
                if len(table) < 2:  # Skip tables with less than 2 rows
                    continue
                    
                df = pd.DataFrame(table[1:], columns=table[0])
                df = df.dropna(how='all').reset_index(drop=True)
                
                if df.empty:
                    continue

                # Find the nearest title
                title = next((t for t in titles), None)
                title_text = title['description'] if title else f"Table_{i}"

                self.tables.append({
                    'title': title_text,
                    'page': page.page_number,
                    'data': df,
                    'filename': f"page_{page.page_number:03d}_table_{i:03d}.xlsx"
                })

        except Exception as e:
            self.logger.error(f"Error processing page {page.page_number}: {str(e)}")

    def process(self):
        with pdfplumber.open(self.pdf_path) as pdf:
            total_pages = len(pdf.pages)
            for i, page in enumerate(pdf.pages, 1):
                self.logger.info(f"Processing page {i}/{total_pages}")
                self._process_page(page)

    def save_tables(self) -> List[str]:
        saved_files = []
        for i, table in enumerate(self.tables, 1):
            try:
                filepath = self.output_dir / table['filename']
                table['data'].to_excel(filepath, index=False)
                saved_files.append(str(filepath))
                self.logger.info(f"Saved {filepath}")
            except Exception as e:
                self.logger.error(f"Error saving table {i}: {e}")
        return saved_files

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract tables from PDF files')
    parser.add_argument('pdf_file', help='Path to PDF file')
    parser.add_argument('-o', '--output', help='Output directory', default='extracted_tables')
    args = parser.parse_args()

    extractor = PDFTableExtractor(args.pdf_file, args.output)
    extractor.process()
    saved_files = extractor.save_tables()
    
    print(f"\nExtracted {len(extractor.tables)} tables")
    print(f"Saved {len(saved_files)} files to: {extractor.output_dir}")

if __name__ == "__main__":
    main()