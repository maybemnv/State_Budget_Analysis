import pdfplumber
import pandas as pd
import re
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Any
import logging
from table_extractor import (
    get_page_elements,
    find_tables,
    reconstruct_table,
    find_table_titles,
    match_tables_to_titles,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def extract_titled_tables_from_pdf(pdf_path: str, add_timestamp: bool = False) -> List[Dict[str, Any]]:
    """
    Extract tables with titles from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        add_timestamp: If True, adds timestamp to output directory to prevent overwrites
    
    Returns:
        List of dictionaries containing extracted table information
    """
    logger.info(f"Starting extraction from: {pdf_path}")
    
    # Create output directory
    output_dir = get_output_directory(pdf_path, add_timestamp)
    logger.info(f"Output directory: {output_dir}")
    
    titled_tables = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                logger.info(f"Processing page {page.page_number}/{len(pdf.pages)}...")
                
                page_elements = get_page_elements(page)
                titles = find_table_titles(page)
                
                if not titles:
                    logger.debug(f"  No table titles found on page {page.page_number}")
                    continue
                
                tables = find_tables(page_elements, page.height)
                if not tables:
                    logger.warning(f"  No tables found on page {page.page_number}")
                    continue
                
                matched_tables = match_tables_to_titles(tables, titles, page.height)
                
                for matched_table in matched_tables:
                    title = matched_table['title']
                    table_bbox = matched_table['table']
                    
                    reconstructed_table = reconstruct_table(table_bbox, page_elements)
                    df = pd.DataFrame(reconstructed_table)
                    
                    filename = sanitize_filename(
                        title['description'],
                        title['number'],
                        page.page_number
                    )
                    
                    output_path = output_dir / f"{filename}.xlsx"
                    
                    try:
                        df.to_excel(output_path, index=False, header=False)
                        logger.info(f"  ✓ Saved: {filename}.xlsx ({df.shape[0]}x{df.shape[1]})")
                    except Exception as excel_error:
                        logger.error(f"  Failed to save Excel: {excel_error}")
                        continue
                    
                    titled_tables.append({
                        'filename': filename,
                        'table_number': title['number'],
                        'description': title['description'],
                        'page': page.page_number,
                        'data': df,
                        'file_path': output_path,
                        'shape': df.shape
                    })
    
    except Exception as e:
        logger.error(f"Fatal error while processing PDF: {e}")
        raise
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Extraction complete!")
    logger.info(f"  Successfully extracted: {len(titled_tables)} tables")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"{ '='*70}\n")
    
    return titled_tables

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

def main():
    """
    Main function with support for command-line arguments.
    
    Usage:
        python extract_tables.py                    # Process all PDFs in current directory
        python extract_tables.py file.pdf           # Process specific PDF file
        python extract_tables.py --timestamp        # Add timestamp to output folders
        python extract_tables.py file.pdf --timestamp
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract titled tables from PDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python extract_tables.py                    # Process all PDFs in current directory
  python extract_tables.py budget.pdf         # Process specific PDF file
  python extract_tables.py --timestamp        # Add timestamp to output folders
  python extract_tables.py budget.pdf -t      # Process specific file with timestamp
        '''
    )
    
    parser.add_argument(
        'pdf_file',
        nargs='?',
        help='Specific PDF file to process (optional, processes all PDFs if not specified)'
    )
    parser.add_argument(
        '-t', '--timestamp',
        action='store_true',
        help='Add timestamp to output directory to prevent overwrites'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Determine which PDF files to process
    if args.pdf_file:
        # Process specific file
        if not os.path.exists(args.pdf_file):
            logger.error(f"Error: File '{args.pdf_file}' not found!")
            return 1
        if not args.pdf_file.lower().endswith('.pdf'):
            logger.error(f"Error: '{args.pdf_file}' is not a PDF file!")
            return 1
        pdf_files = [args.pdf_file]
    else:
        # Process all PDFs in current directory
        pdf_files = [f for f in os.listdir() if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.error("Error: No PDF files found in the current directory!")
            logger.info("Please place PDF files in the same directory as this script.")
            logger.info("Or specify a PDF file: python extract_tables.py <filename.pdf>")
            return 1
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PDF Table Extraction Tool")
    logger.info(f"{ '='*70}")
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.info(f"Timestamp mode: {'ON' if args.timestamp else 'OFF'}")
    logger.info(f"{ '='*70}\n")
    
    # Process each PDF file
    total_tables = 0
    failed_files = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n{'#'*70}")
        logger.info(f"Processing PDF {idx}/{len(pdf_files)}: {pdf_file}")
        logger.info(f"{ '#' * 70}\n")
        
        try:
            # Extract titled tables
            titled_tables = extract_titled_tables_from_pdf(pdf_file, add_timestamp=args.timestamp)
            
            if titled_tables:
                print_extraction_summary(titled_tables)
                total_tables += len(titled_tables)
                logger.info(f"✓ Successfully extracted {len(titled_tables)} titled tables from {pdf_file}")
            else:
                logger.warning(f"⚠ No titled tables found in {pdf_file}")
                logger.info("Make sure the PDF contains tables with titles in format 'Table X.X: Description'")
                
        except Exception as e:
            logger.error(f"✗ Error processing {pdf_file}: {e}")
            failed_files.append(pdf_file)
            if len(pdf_files) > 1:
                logger.info("Continuing with next PDF file...")
            continue
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"Total PDFs processed: {len(pdf_files)}")
    print(f"Total tables extracted: {total_tables}")
    if failed_files:
        print(f"Failed files ({len(failed_files)}): {', '.join(failed_files)}")
    print("=" * 70 + "\n")
    
    return 0 if not failed_files else 1


if __name__ == "__main__":
    sys.exit(main())