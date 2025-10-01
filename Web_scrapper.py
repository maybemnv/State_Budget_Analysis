import requests
from bs4 import BeautifulSoup
import pandas as pd
from time import sleep
from random import randint, uniform
import json
import re
from urllib.parse import urljoin, urlparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompliantCollegeScraper:
    def __init__(self):
        self.base_url = "https://www.collegedunia.com"
        self.session = requests.Session()
        
        # Human-like headers to avoid bot detection
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'no-cache',
            'DNT': '1'  # Do Not Track
        })
        
        self.colleges_data = []
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Respectful delays (longer than before)
        self.min_delay = 3  # Minimum 3 seconds
        self.max_delay = 7  # Maximum 7 seconds
    
    def respectful_delay(self):
        """Add a respectful delay between requests"""
        delay = uniform(self.min_delay, self.max_delay)
        logger.info(f"Waiting {delay:.1f} seconds...")
        sleep(delay)
    
    def get_college_links(self):
        """Get college links from saved HTML file (robots.txt compliant approach)"""
        try:
            logger.info("Loading college links from saved HTML file...")
            with open('sample_page.html', 'r', encoding='utf-8') as f:
                content = f.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            college_links = []
            
            # Find all links
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
                    
                    # Filter out disallowed URLs based on robots.txt
                    parsed = urlparse(full_url)
                    path = parsed.path
                    
                    # Skip if it has disallowed parameters or paths
                    if any(param in href for param in ['?page=', '&page=', '?query=', '&query=', 
                                                     '?slug=', '&slug=', '?course_type=', '&course_type=']):
                        continue
                    
                    # Skip subpages (ranking, reviews, etc.)
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
            
            return college_links[:10]  # Limit for testing
            
        except FileNotFoundError:
            logger.error("sample_page.html not found. Please run analyze_website.py first.")
            return []
        except Exception as e:
            logger.error(f"Error loading college links: {e}")
            return []
    
    def extract_college_info(self, url):
        """Extract college information from a single college page"""
        try:
            logger.info(f"Fetching: {url}")
            
            # Add respectful delay
            self.respectful_delay()
            
            # Make request with timeout
            response = self.session.get(url, timeout=30, allow_redirects=True)
            response.raise_for_status()
            
            # Check if we got valid content
            if len(response.text) < 1000:
                logger.warning(f"Suspiciously small response ({len(response.text)} chars)")
                self.failed_requests += 1
                return None
            
            # Check for common blocking indicators
            if any(indicator in response.text.lower() for indicator in 
                   ['access denied', 'blocked', 'captcha', 'rate limit', 'bot detected']):
                logger.warning("Possible blocking detected in response")
                self.failed_requests += 1
                return None
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Initialize college data
            college = {
                'url': url,
                'scraped_successfully': True
            }
            
            # Extract college name with multiple fallbacks
            name = self.extract_name(soup)
            college['name'] = name
            
            if name == 'N/A':
                logger.warning("Could not extract college name")
                self.failed_requests += 1
                return None
            
            # Extract other information
            college['location'] = self.extract_location(soup)
            college['rating'] = self.extract_rating(soup)
            college['establishment_year'] = self.extract_year(soup)
            college['college_type'] = self.extract_type(soup)
            
            # Extract any additional structured data
            college.update(self.extract_additional_info(soup))
            
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
        strategies = [
            lambda s: s.find('h1'),
            lambda s: s.find('title'),
            lambda s: s.find('meta', {'property': 'og:title'}),
            lambda s: s.select_one('.college-name, .clg-name, .college_name'),
            lambda s: s.select_one('h1[class*="name"], h1[class*="title"]')
        ]
        
        for strategy in strategies:
            try:
                element = strategy(soup)
                if element:
                    if element.name == 'meta':
                        text = element.get('content', '')
                    else:
                        text = element.get_text(strip=True)
                    
                    if text:
                        # Clean up the name
                        text = re.sub(r'\s*[-|]\s*CollegeDunia.*$', '', text, flags=re.IGNORECASE)
                        text = re.sub(r'\s*[-|]\s*Admission.*$', '', text, flags=re.IGNORECASE)
                        return text.strip()
            except:
                continue
        
        return 'N/A'
    
    def extract_location(self, soup):
        """Extract college location"""
        selectors = [
            '.address', '.location', '.city', 
            '[class*="address"]', '[class*="location"]', '[class*="city"]',
            'meta[name="description"]'
        ]
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    if element.name == 'meta':
                        text = element.get('content', '')
                    else:
                        text = element.get_text(strip=True)
                    
                    # Extract location from description if available
                    if 'meta' in selector and text:
                        # Look for location patterns
                        location_match = re.search(r'in ([A-Z][a-z]+(?:\s+[A-Z][a-z]*)*)', text)
                        if location_match:
                            return location_match.group(1)
                    elif text and len(text) < 100:  # Reasonable location length
                        return text
            except:
                continue
        
        return 'N/A'
    
    def extract_rating(self, soup):
        """Extract college rating"""
        selectors = ['.rating', '.score', '[class*="rating"]', '[class*="score"]', '.stars']
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    # Look for rating patterns like "4.5/5" or "4.5"
                    rating_match = re.search(r'(\d+\.?\d*)(?:/\d+)?', text)
                    if rating_match:
                        return rating_match.group(1)
            except:
                continue
        
        return 'N/A'
    
    def extract_year(self, soup):
        """Extract establishment year"""
        text = soup.get_text()
        patterns = [
            r'established\s+(?:in\s+)?(\d{4})',
            r'founded\s+(?:in\s+)?(\d{4})',
            r'since\s+(\d{4})',
            r'(\d{4})\s*-\s*present'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                year = int(match.group(1))
                if 1800 <= year <= 2024:  # Reasonable year range
                    return str(year)
        
        return 'N/A'
    
    def extract_type(self, soup):
        """Extract college type/ownership"""
        selectors = ['[class*="type"]', '[class*="ownership"]', '[class*="affiliation"]']
        
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    text = element.get_text(strip=True)
                    if text and len(text) < 50:
                        return text
            except:
                continue
        
        # Look in text for common types
        text = soup.get_text().lower()
        types = ['government', 'private', 'public', 'autonomous', 'deemed', 'central university']
        for college_type in types:
            if college_type in text:
                return college_type.title()
        
        return 'N/A'
    
    def extract_additional_info(self, soup):
        """Extract any additional structured information"""
        additional = {}
        
        # Look for meta description
        meta_desc = soup.find('meta', {'name': 'description'})
        if meta_desc:
            additional['description'] = meta_desc.get('content', '')[:200] + '...'
        
        return additional
    
    def scrape(self, max_colleges=10):
        """Main scraping function"""
        logger.info("🎓 Compliant College Scraper Started")
        logger.info("=" * 50)
        
        # Get college links
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
        logger.info(f"📊 Success Rate: {self.successful_requests/(self.successful_requests+self.failed_requests)*100:.1f}%")
        
        if self.colleges_data:
            self.save_to_csv()
        else:
            logger.error("❌ No data collected")
    
    def save_to_csv(self, filename='compliant_college_data.csv'):
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
    scraper = CompliantCollegeScraper()
    
    try:
        scraper.scrape(max_colleges=5)  # Start with 5 for testing
    except KeyboardInterrupt:
        logger.info("\n\n⏹️ Scraping interrupted by user")
        if scraper.colleges_data:
            logger.info(f"💾 Saving {len(scraper.colleges_data)} colleges collected so far...")
            scraper.save_to_csv()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        if scraper.colleges_data:
            scraper.save_to_csv()