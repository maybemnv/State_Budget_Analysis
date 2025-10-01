import requests
from bs4 import BeautifulSoup
from time import sleep

def test_link_extraction():
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    session = requests.Session()
    session.headers.update(headers)
    
    url = "https://www.collegedunia.com/engineering-colleges"
    
    print(f"Testing: {url}")
    
    try:
        response = session.get(url, timeout=15)
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)}")
        
        # Check if response is the same as our saved sample
        with open('sample_page.html', 'r', encoding='utf-8') as f:
            saved_content = f.read()
        
        print(f"Saved content length: {len(saved_content)}")
        print(f"Current response same as saved? {response.text == saved_content}")
        
        soup = BeautifulSoup(response.text, 'html.parser')
        all_links = soup.find_all('a', href=True)
        print(f"Total links in current response: {len(all_links)}")
        
        # Save current response for comparison
        with open('current_response.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("✓ Saved current response to current_response.html")
        
        # Test our link extraction logic
        college_links = []
        for link in all_links:
            href = link.get('href')
            if href and ('/university/' in href or '/college/' in href):
                if href.startswith('http'):
                    full_url = href
                elif href.startswith('/'):
                    full_url = "https://www.collegedunia.com" + href
                else:
                    continue
                
                # Filter out specific pages we don't want
                if any(exclude in href for exclude in ['/ranking', '/reviews', '/courses-fees', '/placement', '/admission', '/cutoff']):
                    continue
                    
                college_links.append(full_url)
        
        college_links = list(set(college_links))
        print(f"Found {len(college_links)} unique college links")
        
        if college_links:
            print("Sample college links:")
            for i, link in enumerate(college_links[:5], 1):
                print(f"  {i}. {link}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_link_extraction()