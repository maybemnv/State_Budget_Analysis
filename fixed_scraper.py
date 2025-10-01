import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from random import uniform
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FixedCollegeScraper:
    def __init__(self):
        self.base_url = "https://www.collegedunia.com"
        self.session = requests.Session()
        
        # Headers that work well with compression
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',  # This tells server we accept compression
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'no-cache',
            'DNT': '1'
        })
        
        self.colleges_data = []
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Respectful delays
        self.min_delay = 3
        self.max_delay = 6
    
    def respectful_delay(self):
        """Add a respectful delay between requests"""
        delay = uniform(self.min_delay, self.max_delay)
        logger.info(f"Waiting {delay:.1f} seconds...")
        sleep(delay)
    
    def get_college_links(self):
        """Get college links from saved HTML file"""
        try:
            logger.info("Loading college links from saved HTML file...")
            with open('sample_page.html', 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            college_links = []
            
            all_links = soup.find_all('a', href=True)
            logger.info(f"Found {len(all_links)} total links")
            
            for link in all_links:
                href = link.get('href')
                if not href:
                    continue
                
                # Check if it's a college or university page
                if '/university/' in href or '/college/' in href:
                    # Convert to absolute URL
                    if href.startswith('http'):
                        full_url = href
                    elif href.startswith('/'):
                        full_url = urljoin(self.base_url, href)
                    else:
                        continue
                    
                    # Skip subpages and parameterized URLs
                    parsed = urlparse(full_url)
                    path = parsed.path
                    
                    # Skip if it has query parameters (robots.txt compliance)
                    if parsed.query:
                        continue
                    
                    # Skip subpages
                    if any(subpage in path for subpage in ['/ranking', '/reviews', '/courses-fees', 
                                                          '/placement', '/admission', '/cutoff']):
                        continue
                    
                    # Only main college pages (simple path structure)
                    path_parts = [p for p in path.split('/') if p]
                    if len(path_parts) == 2:  # e.g., ['university', 'college-id-name']
                        college_links.append(full_url)
            
            # Remove duplicates
            college_links = list(set(college_links))
            logger.info(f"Found {len(college_links)} valid college links")
            
            return college_links[:8]  # Test with 8 colleges
            
        except FileNotFoundError:
            logger.error("sample_page.html not found. Please run analyze_website.py first.")
            return []
        except Exception as e:
            logger.error(f"Error loading college links: {e}")
            return []
    
    def extract_college_info(self, url):
        """Extract college information from a single college page with proper decompression"""
        try:
            logger.info(f"Fetching: {url}")
            
            # Add respectful delay
            self.respectful_delay()
            
            # Make request - requests library should automatically decompress
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Double-check the response was properly decoded
            if response.encoding is None:
                response.encoding = 'utf-8'
            
            # Get the text content (should be automatically decompressed)
            html_content = response.text
            
            # Verify we got proper HTML content
            if len(html_content) < 1000:
                logger.warning(f"Response too small ({len(html_content)} chars)")
                self.failed_requests += 1
                return None
            
            # Check if content looks like HTML (not compressed binary)
            if not html_content.strip().startswith('<'):
                logger.warning("Response doesn't look like HTML")
                self.failed_requests += 1
                return None
            
            # Check for blocking indicators
            if any(indicator in html_content.lower() for indicator in 
                   ['access denied', 'blocked', 'captcha', 'rate limit', 'bot detected']):
                logger.warning("Possible blocking detected")
                self.failed_requests += 1
                return None
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Initialize college data
            college = {
                'url': url,
                'response_size': len(html_content)
            }
            
            # Extract college name
            name = self.extract_name(soup)
            college['name'] = name
            
            if name == 'N/A':
                logger.warning("Could not extract college name")
                # Don't fail completely, continue with other fields
            
            # Extract other information
            college['location'] = self.extract_location(soup)
            college['rating'] = self.extract_rating(soup)
            college['establishment_year'] = self.extract_year(soup)
            college['college_type'] = self.extract_type(soup)
            college['description'] = self.extract_description(soup)
            
            self.successful_requests += 1
            logger.info(f"✅ Successfully scraped: {name}")
            return college
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error for {url}: {e}")
            self.failed_requests += 1
            return None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {e}")
            self.failed_requests += 1
            return None
    
    def extract_name(self, soup):
        """Extract college name using multiple strategies"""
        # Strategy 1: Page title
        title_elem = soup.find('title')
        if title_elem:
            title_text = title_elem.get_text(strip=True)
            if title_text:
                # Clean up the title
                name = re.sub(r'\s*[-|]\s*CollegeDunia.*$', '', title_text, flags=re.IGNORECASE)
                name = re.sub(r'\s*[-|]\s*Admission.*$', '', name, flags=re.IGNORECASE)
                name = re.sub(r'\s*[-|]\s*Fees.*$', '', name, flags=re.IGNORECASE)
                if name.strip() and len(name.strip()) > 5:
                    return name.strip()
        
        # Strategy 2: Meta og:title
        og_title = soup.find('meta', {'property': 'og:title'})
        if og_title:
            og_text = og_title.get('content', '')
            if og_text:
                name = re.sub(r'\s*[-|]\s*CollegeDunia.*$', '', og_text, flags=re.IGNORECASE)
                if name.strip() and len(name.strip()) > 5:
                    return name.strip()
        
        # Strategy 3: H1 tags
        h1_tags = soup.find_all('h1')
        for h1 in h1_tags:
            h1_text = h1.get_text(strip=True)
            if h1_text and len(h1_text) > 5 and len(h1_text) < 200:
                return h1_text
        
        # Strategy 4: Common class names
        name_selectors = ['.college-name', '.clg-name', '.college_name', '.university-name']
        for selector in name_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                if text and len(text) > 5:
                    return text
        
        # Strategy 5: Extract from URL as fallback
        url_part = soup.find('link', {'rel': 'canonical'})
        if url_part:
            url = url_part.get('href', '')
        else:
            url = getattr(soup, 'url', '')
        
        if url:
            # Extract from URL pattern like /university/25914-vellore-institute-of-technology
            match = re.search(r'/(?:university|college)/\d+-(.*)', url)
            if match:
                name_from_url = match.group(1).replace('-', ' ').title()
                return name_from_url
        
        return 'N/A'
    
    def extract_location(self, soup):
        """Extract college location"""
        # Look in meta description first
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            desc = meta_desc.get('content', '')
            # Look for location patterns like "in Mumbai", "located in Delhi"
            location_patterns = [
                r'\bin ([A-Z][a-zA-Z\s]+)(?:,|\.|$)',
                r'located in ([A-Z][a-zA-Z\s]+)(?:,|\.|$)',
                r'situated in ([A-Z][a-zA-Z\s]+)(?:,|\.|$)'
            ]
            for pattern in location_patterns:
                match = re.search(pattern, desc)
                if match:
                    location = match.group(1).strip()
                    if len(location) < 50:  # Reasonable location length
                        return location
        
        # Look for address elements
        address_selectors = ['.address', '.location', '[class*="address"]', '[class*="location"]']
        for selector in address_selectors:
            elem = soup.select_one(selector)
            if elem:
                text = elem.get_text(strip=True)
                if text and len(text) < 100:
                    return text
        
        return 'N/A'
    
    def extract_rating(self, soup):
        """Extract college rating"""
        # Look for rating patterns in text
        text = soup.get_text()
        rating_patterns = [
            r'rating[:\s]*(\d+\.?\d*)(?:/\d+)?',
            r'score[:\s]*(\d+\.?\d*)(?:/\d+)?',
            r'(\d+\.\d+)\s*(?:out of|/)\s*\d+',
            r'(\d+\.\d+)\s*stars?'
        ]
        
        for pattern in rating_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                rating = float(match.group(1))
                if 0 <= rating <= 5:  # Reasonable rating range
                    return str(rating)
        
        return 'N/A'
    
    def extract_year(self, soup):
        """Extract establishment year"""
        text = soup.get_text()
        patterns = [
            r'established\s+(?:in\s+)?(\d{4})',
            r'founded\s+(?:in\s+)?(\d{4})',
            r'since\s+(\d{4})',
            r'started\s+(?:in\s+)?(\d{4})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 1800 <= year <= 2024:
                    return str(year)
        
        return 'N/A'
    
    def extract_type(self, soup):
        """Extract college type"""
        text = soup.get_text().lower()
        types = [
            ('government', 'Government'),
            ('private', 'Private'), 
            ('public', 'Public'),
            ('autonomous', 'Autonomous'),
            ('deemed university', 'Deemed University'),
            ('central university', 'Central University'),
            ('state university', 'State University')
        ]
        
        for search_term, display_name in types:
            if search_term in text:
                return display_name
        
        return 'N/A'
    
    def extract_description(self, soup):
        """Extract meta description"""
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            desc = meta_desc.get('content', '')
            return desc[:300] + '...' if len(desc) > 300 else desc
        
        return 'N/A'
    
    def scrape(self, max_colleges=8):
        """Main scraping function"""
        logger.info("🎓 Fixed College Scraper Started")
        logger.info("=" * 50)
        
        college_links = self.get_college_links()
        
        if not college_links:
            logger.error("❌ No college links found. Exiting.")
            return
        
        logger.info(f"📋 Found {len(college_links)} college links")
        logger.info(f"🎯 Will process first {min(max_colleges, len(college_links))} colleges")
        logger.info(f"⏱️  Using delays of {self.min_delay}-{self.max_delay} seconds between requests")
        
        # Process each college
        for i, url in enumerate(college_links[:max_colleges], 1):
            logger.info(f"\n[{i}/{min(max_colleges, len(college_links))}] Processing college...")
            
            college_data = self.extract_college_info(url)
            
            if college_data:
                self.colleges_data.append(college_data)
        
        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("🎉 Scraping Summary:")
        logger.info(f"✅ Successful: {self.successful_requests}")
        logger.info(f"❌ Failed: {self.failed_requests}")
        
        if self.successful_requests + self.failed_requests > 0:
            success_rate = self.successful_requests/(self.successful_requests+self.failed_requests)*100
            logger.info(f"📊 Success Rate: {success_rate:.1f}%")
        
        if self.colleges_data:
            self.save_to_csv()
        else:
            logger.error("❌ No data collected")
    
    def save_to_csv(self, filename='fixed_college_data.csv'):
        """Save scraped data to CSV"""
        if self.colleges_data:
            try:
                df = pd.DataFrame(self.colleges_data)
                df.to_csv(filename, index=False, encoding='utf-8')
                logger.info(f"\n💾 Data saved to {filename}")
                logger.info(f"📋 Columns: {list(df.columns)}")
                logger.info(f"📊 Rows: {len(df)}")
                logger.info("\n📋 Sample data:")
                print(df.head(3).to_string(index=False))
            except Exception as e:
                logger.error(f"❌ Error saving to CSV: {e}")

if __name__ == "__main__":
    scraper = FixedCollegeScraper()
    
    try:
        scraper.scrape(max_colleges=5)  # Test with 5 colleges
    except KeyboardInterrupt:
        logger.info("\n\n⏹️ Scraping interrupted by user")
        if scraper.colleges_data:
            logger.info(f"💾 Saving {len(scraper.colleges_data)} colleges collected so far...")
            scraper.save_to_csv()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if scraper.colleges_data:
            scraper.save_to_csv()