import requests
from bs4 import BeautifulSoup
import os
import time
import logging
from typing import Dict, Any, List, Optional

class F1WikiScraper:
    """Wikipedia scraping engine with rate limiting and caching"""
    
    def __init__(self, cache_dir='wiki_cache'):
        self.base_url = "https://en.wikipedia.org/wiki/"
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.headers = {
            'User-Agent': 'F1DataBot/1.0 (+https://github.com/arnavxox/f1-analysis)'
        }
        self.logger = logging.getLogger(__name__)
        
    def _get_cached_page(self, page_name: str) -> str:
        """Retrieve cached page or download fresh copy"""
        cache_path = os.path.join(self.cache_dir, f"{page_name}.html")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    self.logger.info(f"Using cached page for {page_name}")
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Error reading cache for {page_name}: {e}")
                # Continue to fetch fresh copy if cache read fails
        
        time.sleep(1.5)  # Respect rate limits
        try:
            self.logger.info(f"Fetching page from Wikipedia: {page_name}")
            response = requests.get(f"{self.base_url}{page_name}", headers=self.headers)
            if response.status_code == 200:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                return response.text
            else:
                self.logger.error(f"Failed to fetch page: {response.status_code}")
                return ""
        except Exception as e:
            self.logger.error(f"Error fetching page {page_name}: {e}")
            return ""

    def _try_multiple_page_formats(self, year: int, race: str) -> str:
        """Try different page naming formats to find the correct Wikipedia page"""
        # List of possible formats
        formats = [
            f"{year}_{race.replace(' ', '_')}_Grand_Prix",  # Standard format
            f"{race.replace(' ', '_')}_{year}_Formula_One_Grand_Prix",
            f"{race.replace(' ', '_')}_Grand_Prix_{year}",
            f"{year}_Formula_One_World_Championship_Race_{race.replace(' ', '_')}"
        ]
        
        # Try each format
        for page_format in formats:
            html = self._get_cached_page(page_format)
            if html:
                soup = BeautifulSoup(html, 'html.parser')
                # Check if this looks like a valid F1 race page
                if soup.find('table', {'class': 'infobox vevent'}) or "Grand Prix" in soup.title.text:
                    return html
        
        # If all formats fail, try search
        search_query = f"{race} {year} Grand Prix Formula One"
        search_query = search_query.replace(' ', '_')
        return self._get_cached_page(search_query)

    def get_race_summary(self, year: int, race: str) -> Dict[str, str]:
        """Extract race summary from Wikipedia"""
        try:
            # Try multiple page formats
            html = self._try_multiple_page_formats(year, race)
            if not html:
                return {"Error": f"Could not find Wikipedia page for {race} {year}"}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Find infobox - try different classes
            infobox = None
            for class_name in ['infobox vevent', 'infobox', 'infobox racing']:
                infobox = soup.find('table', {'class': class_name})
                if infobox:
                    break
            
            if not infobox:
                return {"Error": f"No race information found for {race} {year}"}
            
            # Extract data from infobox
            data = {}
            for row in infobox.find_all('tr'):
                header = row.find('th')
                value = row.find('td')
                
                if header and value:
                    key = header.get_text(strip=True)
                    val = value.get_text(strip=True)
                    
                    # Clean up key and value
                    key = key.replace('\n', ' ').strip()
                    val = val.replace('\n', ' ').strip()
                    
                    # Store in dictionary
                    data[key] = val
            
            # Add some standard fields if missing
            if 'Race name' not in data and soup.title:
                data['Race name'] = soup.title.text.split(' - ')[0].strip()
            
            if 'Date' not in data:
                date_tag = soup.find('span', {'class': 'bday dtstart published updated'})
                if date_tag:
                    data['Date'] = date_tag.text
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error extracting race summary: {e}")
            return {"Error": f"Failed to extract data: {str(e)}"}

    def get_driver_profile(self, driver_name: str) -> Dict[str, str]:
        """Get driver biography and career stats"""
        try:
            # Try both driver name formats
            html = ""
            for name_format in [driver_name, f"Formula_One_driver_{driver_name}"]:
                page_slug = name_format.replace(' ', '_')
                html = self._get_cached_page(page_slug)
                if html:
                    break
            
            if not html:
                return {"Error": f"Could not find Wikipedia page for {driver_name}"}
                
            soup = BeautifulSoup(html, 'html.parser')
            
            profile = {}
            infobox = soup.find('table', {'class': 'infobox'})
            if infobox:
                for row in infobox.find_all('tr'):
                    header = row.find('th')
                    value = row.find('td')
                    
                    if header and value:
                        key = header.get_text(strip=True)
                        val = value.get_text(strip=True)
                        
                        # Clean up and store
                        key = key.replace('\n', ' ').strip()
                        val = val.replace('\n', ' ').strip()
                        profile[key] = val
            
            # Add name if missing
            if 'Name' not in profile and soup.title:
                profile['Name'] = soup.title.text.split(' - ')[0].strip()
                
            return profile
            
        except Exception as e:
            self.logger.error(f"Error extracting driver profile: {e}")
            return {"Error": f"Failed to extract driver data: {str(e)}"}
