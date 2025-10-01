import requests
from bs4 import BeautifulSoup
import re

def analyze_website():
    # Fetch the page and check its structure
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }

    url = 'https://www.collegedunia.com/engineering-colleges'
    
    try:
        print(f"Analyzing: {url}")
        response = requests.get(url, headers=headers, timeout=15)
        print(f'Status Code: {response.status_code}')
        print(f'Content Length: {len(response.text)}')

        soup = BeautifulSoup(response.text, 'html.parser')

        # Look for different patterns that might contain college links
        print('\n=== Analyzing page structure ===')

        # Check all links on the page
        all_links = soup.find_all('a', href=True)
        print(f'Total links found: {len(all_links)}')

        # Filter for college-related links
        college_links = [link for link in all_links if '/college' in link.get('href', '')]
        print(f'Links containing "/college": {len(college_links)}')

        if college_links:
            print('Sample college links:')
            for link in college_links[:5]:
                href = link.get('href')
                text = link.get_text(strip=True)
                print(f'  - {href} ({text[:50]}...)')

        # Check for common college listing patterns
        patterns_to_check = ['college', 'engineering', 'institute', 'university']
        for pattern in patterns_to_check:
            matching_links = [link for link in all_links if pattern.lower() in link.get('href', '').lower()]
            print(f'Links containing "{pattern}": {len(matching_links)}')

        # Look for any divs or containers that might hold college information
        print('\n=== Looking for college containers ===')
        
        # Try to find common class names that might contain college info
        potential_classes = ['college', 'card', 'item', 'list', 'result', 'institution']
        for class_name in potential_classes:
            elements = soup.find_all(attrs={"class": re.compile(class_name, re.I)})
            if elements:
                print(f'Elements with class containing "{class_name}": {len(elements)}')
                if elements:
                    first_elem = elements[0]
                    print(f'  Sample: {first_elem.get("class")}')

        # Check if the page might be using JavaScript to load content
        scripts = soup.find_all('script')
        js_indicators = ['ajax', 'fetch', 'xhr', 'api', 'json']
        js_content = False
        for script in scripts:
            if script.string:
                for indicator in js_indicators:
                    if indicator in script.string.lower():
                        js_content = True
                        break
                if js_content:
                    break
        
        if js_content:
            print('\n⚠️  This page appears to use JavaScript to load content dynamically.')
            print('   You might need to use Selenium or similar tools for scraping.')

        # Save a sample of the HTML for manual inspection
        with open('sample_page.html', 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f'\n✓ Saved sample HTML to sample_page.html for manual inspection')

    except Exception as e:
        print(f'Error analyzing website: {e}')

if __name__ == "__main__":
    analyze_website()