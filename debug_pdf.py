import pdfplumber
import sys
from table_extractor import find_table_titles, extract_tables_from_page
def debug_pdf(pdf_path: str):
    """
    Prints debugging information about the tables and titles on each page of a PDF.
    """
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            print(f"--- Page {page.page_number} ---")
            titles = find_table_titles(page)
            tables = extract_tables_from_page(page)
            print(f"Found {len(titles)} titles:")
            for title in titles:
                print(f"  - {title['number']}: {title['description']} (bbox: {title['bbox']})")
            print(f"Found {len(tables)} tables:")
            for i, table in enumerate(tables):
                print(f"  - Table {i+1} (bbox: {table.bbox})")
if __name__ == "__main__":
    if len(sys.argv) > 1:
        debug_pdf(sys.argv[1])
    else:
        print("Usage: python debug_pdf.py <pdf_path>")