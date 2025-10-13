import requests
from bs4 import BeautifulSoup
import re

def test_single_college_page():
    # Test a single college page
    url = "https://collegedunia.com/university/25914-vellore-institute-of-technology-vit-university-vellore"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }
    
    try:
        print(f"Testing: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        print(f"Status Code: {response.status_code}")
        print(f"Content Length: {len(response.text)}")
        print(f"Response Headers: {dict(response.headers)}")
        
        # Save response for analysis
        with open('test_college_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print("✅ Saved response to test_college_page.html")
        
        # Quick analysis
        soup = BeautifulSoup(response.text, 'html.parser')
        
        print("\n=== Quick Analysis ===")
        
        # Check title
        title = soup.find('title')
        if title:
            print(f"Title: {title.get_text()}")
        
        # Check for h1 tags
        h1_tags = soup.find_all('h1')
        print(f"H1 tags found: {len(h1_tags)}")
        for i, h1 in enumerate(h1_tags[:3]):
            print(f"  H1 {i+1}: {h1.get_text(strip=True)[:100]}...")
        
        # Check for meta og:title
        og_title = soup.find('meta', {'property': 'og:title'})
        if og_title:
            print(f"OG Title: {og_title.get('content')}")
        
        # Check if page seems to be working
        if 'vellore institute of technology' in response.text.lower():
            print("✅ Page contains expected content")
        else:
            print("❌ Page doesn't contain expected content")
        
        # Check for blocking indicators
        blocking_indicators = ['access denied', 'blocked', 'captcha', 'rate limit', 'bot detected', 'cloudflare']
        for indicator in blocking_indicators:
            if indicator in response.text.lower():
                print(f"⚠️  Found blocking indicator: {indicator}")
        
        # Check first 1000 chars
        print(f"\nFirst 500 chars of response:")
        print("=" * 50)
        print(response.text[:500])
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_single_college_page()