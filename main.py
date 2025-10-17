
import pdfplumber
import pandas as pd
import sys
import logging
import os
from pathlib import Path
from table_extractor import (
    find_table_titles,
    extract_tables_from_page,
    match_tables_to_titles,
    clean_and_validate_table,
)
from utils import (
    sanitize_filename,
    get_output_directory,
    print_extraction_summary,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_titled_tables_from_pdf(pdf_path: str, add_timestamp: bool = False) -> list[dict[str, any]]:
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
                
                titles = find_table_titles(page)
                if not titles:
                    logger.debug(f"  No table titles found on page {page.page_number}")
                    continue
                
                tables = extract_tables_from_page(page)
                if not tables:
                    logger.warning(f"  No tables extracted on page {page.page_number} despite finding {len(titles)} title(s)")
                    continue
                
                matched_tables = match_tables_to_titles(tables, titles, page.height)
                
                for matched_table in matched_tables:
                    title = matched_table['title']
                    table = matched_table['table']
                    
                    cleaned_df = clean_and_validate_table(table.extract())
                    
                    if cleaned_df is None:
                        logger.warning(f"  Table on page {page.page_number} skipped after cleaning.")
                        continue
                    
                    filename = sanitize_filename(
                        title['description'],
                        title['number'],
                        page.page_number
                    )
                    
                    output_path = output_dir / f"{filename}.xlsx"
                    
                    try:
                        cleaned_df.to_excel(output_path, index=False)
                        logger.info(f"  ✓ Saved: {filename}.xlsx ({cleaned_df.shape[0]}x{cleaned_df.shape[1]})")
                    except Exception as excel_error:
                        logger.error(f"  Failed to save Excel: {excel_error}")
                        continue
                    
                    titled_tables.append({
                        'filename': filename,
                        'table_number': title['number'],
                        'description': title['description'],
                        'page': page.page_number,
                        'data': cleaned_df,
                        'file_path': output_path,
                        'shape': cleaned_df.shape
                    })
    
    except Exception as e:
        logger.error(f"Fatal error while processing PDF: {e}")
        raise
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Extraction complete!")
    logger.info(f"  Successfully extracted: {len(titled_tables)} tables")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"{'='*70}\n")
    
    return titled_tables

def main():
    """
    Main function with support for command-line arguments.
    
    Usage:
        python main.py                    # Process all PDFs in current directory
        python main.py file.pdf           # Process specific PDF file
        python main.py --timestamp        # Add timestamp to output folders
        python main.py file.pdf --timestamp
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract titled tables from PDF files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py                    # Process all PDFs in current directory
  python main.py budget.pdf         # Process specific PDF file
  python main.py --timestamp        # Add timestamp to output folders
  python main.py budget.pdf -t      # Process specific file with timestamp
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
            logger.info("Or specify a PDF file: python main.py <filename.pdf>")
            return 1
    
    logger.info(f"\n{'='*70}")
    logger.info(f"PDF Table Extraction Tool")
    logger.info(f"{'='*70}")
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    logger.info(f"Timestamp mode: {'ON' if args.timestamp else 'OFF'}")
    logger.info(f"{'='*70}\n")
    
    # Process each PDF file
    total_tables = 0
    failed_files = []
    
    for idx, pdf_file in enumerate(pdf_files, 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing PDF {idx}/{len(pdf_files)}: {pdf_file}")
        logger.info(f"\n{'='*70}")
        
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
