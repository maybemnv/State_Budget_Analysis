import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from random import randint
import json
import re

class CollegeDuniaScraper:
    def __init__(self):
        self.base_url = "https://www.collegedunia.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        self.colleges_data = []
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def get_college_links(self, page=1):
        """Fetch college links from search results page"""
        # Updated URL structure for CollegeDunia
        url = f"{self.base_url}/engineering-colleges"
        
        # Add page parameter if page > 1
        if page > 1:
            url += f"?page={page}"
        
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract college/university links from the page
            college_links = []
            
            # Find all links on the page
            all_links = soup.find_all('a', href=True)
            print(f"Total links found: {len(all_links)}")
            
            # Filter for college and university links
            for link in all_links:
                href = link.get('href')
                if href and ('/university/' in href or '/college/' in href):
                    # Build full URL
                    if href.startswith('http'):
                        full_url = href
                    elif href.startswith('/'):
                        full_url = self.base_url + href
                    else:
                        continue
                    
                    # Filter out specific pages we don't want (rankings, reviews, etc.)
                    if any(exclude in href for exclude in ['/ranking', '/reviews', '/courses-fees', '/placement', '/admission', '/cutoff']):
                        continue
                        
                    college_links.append(full_url)
            
            # Remove duplicates
            college_links = list(set(college_links))
            
            # Filter to main college pages only (not subpages)
            main_college_links = []
            for link in college_links:
                # Check if this is a main college page (ends with college name, not a subpage)
                if link.count('/') <= 4:  # Base structure: https://domain.com/university/id-name or https://domain.com/college/id-name
                    main_college_links.append(link)
            
            print(f"Found {len(college_links)} total college-related links")
            print(f"Found {len(main_college_links)} main college pages")
            
            # Return a reasonable sample for testing
            return main_college_links[:15]  # Limit to first 15 for testing
            
        except Exception as e:
            print(f"Error fetching college links from {url}: {e}")
            return []
    
    def get_college_details(self, url):
        """Extract college details from individual college page"""
        try:
            print(f"Fetching details from: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract college details
            college = {'url': url}
            
            # College name - try multiple selectors
            name_selectors = ['h1', '.college-name', '.clg-name', 'h1.college_name', '.college_name']
            college['name'] = self._extract_text_by_selectors(soup, name_selectors) or 'N/A'
            
            # Location - try multiple selectors
            location_selectors = ['.college_address', '.location', '.address', '[class*="address"]', '[class*="location"]']
            college['location'] = self._extract_text_by_selectors(soup, location_selectors) or 'N/A'
            
            # Ratings - try multiple selectors
            rating_selectors = ['.rating', '.star-rating', '[class*="rating"]', '.score']
            college['rating'] = self._extract_text_by_selectors(soup, rating_selectors) or 'N/A'
            
            # Try to extract establishment year
            year_selectors = ['[class*="establish"]', '[class*="year"]']
            college['established'] = self._extract_text_by_selectors(soup, year_selectors) or 'N/A'
            
            # Try to extract college type
            type_selectors = ['[class*="type"]', '[class*="ownership"]']
            college['type'] = self._extract_text_by_selectors(soup, type_selectors) or 'N/A'
            
            # Extract any additional structured data
            detail_elements = soup.find_all('div', class_='detail-item')
            for item in detail_elements:
                key = item.find('div', class_='detail-label')
                value = item.find('div', class_='detail-value')
                if key and value:
                    college[key.text.strip().lower().replace(' ', '_')] = value.text.strip()
            
            return college
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
    
    def _extract_text_by_selectors(self, soup, selectors):
        """Helper method to try multiple CSS selectors and return first match"""
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element and element.text.strip():
                    return element.text.strip()
            except:
                continue
        return None
    
    def scrape(self, max_pages=2):
        """Main scraping function"""
        print(f"Starting scrape with max_pages={max_pages}")
        
        for page in range(1, max_pages + 1):
            print(f"\n=== Scraping page {page} ===")
            college_links = self.get_college_links(page)
            
            if not college_links:
                print(f"No college links found on page {page}. Stopping.")
                break
            
            print(f"Processing {len(college_links)} colleges from page {page}")
            
            successful_scrapes = 0
            for i, link in enumerate(college_links, 1):
                print(f"\n[{i}/{len(college_links)}] Processing: {link}")
                
                try:
                    college_data = self.get_college_details(link)
                    if college_data and college_data.get('name') != 'N/A':
                        self.colleges_data.append(college_data)
                        successful_scrapes += 1
                        print(f"✓ Successfully scraped: {college_data.get('name', 'Unknown')}")
                    else:
                        print(f"✗ Failed to extract valid data from {link}")
                        
                except Exception as e:
                    print(f"✗ Error processing {link}: {e}")
                
                # Add random delay to avoid getting blocked
                sleep(randint(2, 4))
            
            print(f"\nPage {page} summary: {successful_scrapes}/{len(college_links)} colleges successfully scraped")
            
            # Break if we got no successful scrapes (might indicate structure change)
            if successful_scrapes == 0:
                print("No successful scrapes on this page. Website structure might have changed.")
                break
        
        print(f"\n=== Scraping completed ===")
        print(f"Total colleges scraped: {len(self.colleges_data)}")
        
        # Save data to CSV if we have any data
        if self.colleges_data:
            self.save_to_csv()
        else:
            print("No data collected. Please check the website structure or try a different approach.")
    
    def save_to_csv(self, filename='college_data.csv'):
        """Save scraped data to CSV"""
        if self.colleges_data:
            try:
                df = pd.DataFrame(self.colleges_data)
                df.to_csv(filename, index=False, encoding='utf-8')
                print(f"\n✓ Data saved to {filename}")
                print(f"  Columns: {list(df.columns)}")
                print(f"  Rows: {len(df)}")
                print(f"  Sample data:")
                print(df.head(3).to_string(index=False))
            except Exception as e:
                print(f"\n✗ Error saving to CSV: {e}")
        else:
            print("\n✗ No data to save")

if __name__ == "__main__":
    print("CollegeDunia Web Scraper")
    print("=" * 25)
    
    scraper = CollegeDuniaScraper()
    
    # Test with 1 page first
    try:
        scraper.scrape(max_pages=1)
    except KeyboardInterrupt:
        print("\n\nScraping interrupted by user")
        if scraper.colleges_data:
            print(f"Saving {len(scraper.colleges_data)} colleges collected so far...")
            scraper.save_to_csv()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        if scraper.colleges_data:
            print(f"Saving {len(scraper.colleges_data)} colleges collected so far...")
            scraper.save_to_csv()
