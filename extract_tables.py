import pdfplumber
import pandas as pd
import re
import os
from pathlib import Path

def find_table_title(text, page_num):
    """
    Find table titles in the format 'Table X.X: [Description]'
    Returns the table number and description if found, None otherwise
    """
    # Pattern to match table titles like "Table 4.2: Revenue Analysis"
    pattern = r'Table\s+(\d+\.\d+):\s*(.+)'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        table_num = match.group(1)
        description = match.group(2).strip()
        return {
            'number': table_num,
            'description': description,
            'page': page_num
        }
    return None


def clean_dataframe(df):
    """Clean and validate DataFrame"""
    if df is None or df.empty:
        return None
    
    # Replace empty strings with None
    df = df.replace('', None)
    
    # Remove completely empty rows and columns
    df = df.dropna(how='all')
    df = df.dropna(axis=1, how='all')
    
    # Skip if table is too small (less than 2 rows or 2 columns)
    if df.shape[0] < 2 or df.shape[1] < 2:
        return None
    
    return df


def extract_titled_tables_from_pdf(pdf_path):

    # Create output directory for titled tables
    output_dir = Path('extracted_tables')
    output_dir.mkdir(exist_ok=True)
    
    titled_tables = []
    
    print("Scanning PDF for titled tables...")
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            print(f"Processing page {page_num}...")
            
            # Extract text from page to find table titles
            page_text = page.extract_text() or ""
            
            # Find all table titles on this page
            table_titles = []
            for match in re.finditer(r'Table\s+(\d+\.\d+):\s*(.+?)(?=\n|$)', page_text, re.IGNORECASE | re.MULTILINE):
                table_titles.append({
                    'number': match.group(1),
                    'description': match.group(2).strip(),
                    'page': page_num
                })
            
            if not table_titles:
                continue
                
            print(f"  Found {len(table_titles)} table title(s) on page {page_num}")
            
            # Extract tables from the page
            tables = page.extract_tables()
            
            if tables:
                # For each table found, check if it corresponds to a titled table
                for table_idx, table in enumerate(tables):
                    if not table or len(table) < 2:
                        continue
                    
                    # Convert to DataFrame
                    try:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df = clean_dataframe(df)
                        
                        if df is not None:
                            # Find the closest table title (assuming tables appear after their titles)
                            table_title = table_titles[min(table_idx, len(table_titles) - 1)]
                            
                            # Create clean filename
                            safe_description = re.sub(r'[^\w\s-]', '', table_title['description'])
                            safe_description = re.sub(r'\s+', '_', safe_description.strip())
                            filename = f"Table_{table_title['number'].replace('.', '_')}_page_{page_num}_{safe_description}"
                            
                            # Save to CSV
                            output_path = output_dir / f"{filename}.csv"
                            df.to_csv(output_path, index=False, encoding='utf-8-sig')
                            
                            print(f"  Saved: {output_path}")
                            
                            titled_tables.append({
                                'filename': filename,
                                'table_number': table_title['number'],
                                'description': table_title['description'],
                                'page': page_num,
                                'data': df,
                                'file_path': output_path
                            })
                            
                    except Exception as e:
                        print(f"  Warning: Error processing table on page {page_num}: {e}")
                        continue
    
    print(f"\nExtraction complete! Found {len(titled_tables)} titled tables.")
    return titled_tables
def main():
    # Path to your PDF file
    pdf_file = "Report-No.-1-of-2025-English-SFAR-2023-24-068c11ca17d08b3.79273636.pdf"
    
    # Check if file exists
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file '{pdf_file}' not found!")
        print("Please place the PDF file in the same directory as this script.")
        return
    
    try:
        # Extract titled tables
        titled_tables = extract_titled_tables_from_pdf(pdf_file)
        
        if titled_tables:
            print("\n" + "=" * 70)
            print("EXTRACTION SUMMARY")
            print("=" * 70)
            
            for table in titled_tables:
                print(f"Table {table['table_number']}: {table['description']}")
                print(f"   Page {table['page']} | File: {table['filename']}.csv")
                print(f"   Size: {table['data'].shape[0]} rows x {table['data'].shape[1]} columns")
                print()
            
            print(f"Successfully extracted {len(titled_tables)} titled tables!")
        else:
            print("\nWarning: No titled tables found in the PDF.")
            print("Make sure the PDF contains tables with titles in format 'Table X.X: Description'")
            
    except Exception as e:
        print(f"\nError during extraction: {e}")
        print("Please check the PDF file and try again.")


if __name__ == "__main__":
    main()