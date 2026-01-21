import streamlit as st
import aiohttp
import asyncio
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
import json
import re
from bs4 import BeautifulSoup
from typing import List, Dict, Optional, Tuple

from models import ClimateData, ListingItem, AnalysisResponse

# --- Constants ---
EXCLUDED_TITLE_KEYWORDS = [
    "market report", "buyers guide", "buyer's guide", 
    "news article", "blog post"
]

class GeocodingService:
    """Handles reverse geocoding to find the nearest town/city."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "FarmScout_App/1.0"
        }

    def get_location_name(self, lat: float, lon: float) -> str:
        """
        Get address from coordinates. 
        If exact location has no town, search for nearest populated place.
        """
        try:
            # 1. Try exact reverse geocoding
            url = "https://nominatim.openstreetmap.org/reverse"
            params = {
                "lat": lat, "lon": lon, "format": "json"
            }
            response = requests.get(url, params=params, headers=self.headers, timeout=5)
            data = response.json()
            
            address = data.get("address", {})
            town = (address.get("town") or address.get("city") or 
                   address.get("village") or address.get("hamlet"))
            state = address.get("state")
            
            if town:
                return f"{town}, {state}" if state else town
            
            # 2. If no town, find nearest
            return self._find_nearest_town(lat, lon, state)
            
        except Exception:
            return f"Region near {lat:.3f}, {lon:.3f}"

    def _find_nearest_town(self, lat, lon, state):
        """Search for nearby towns in expanding radius."""
        search_url = "https://nominatim.openstreetmap.org/search"
        search_radii = [0.1, 0.5, 1.0]  # ~11km to ~110km
        
        for radius in search_radii:
            viewbox = f"{lon-radius},{lat+radius},{lon+radius},{lat-radius}"
            params = {
                "format": "json", "bounded": 1, "viewbox": viewbox,
                "limit": 1, "featuretype": "city,town,village"
            }
            try:
                # Add state context if available to prioritize local results
                params["q"] = f"town in {state}" if state else "town"
                
                resp = requests.get(search_url, params=params, headers=self.headers, timeout=5)
                results = resp.json()
                if results:
                    name = results[0].get("display_name", "").split(",")[0]
                    return f"{name}, {state}" if state else name
            except:
                continue
                
        return f"Region near {lat:.3f}, {lon:.3f}"

class AsyncScraper:
    """Handles asynchronous web scraping with concurrency limits."""
    
    def __init__(self, max_concurrency: int = 10):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

    async def fetch_text(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """Fetch and extract text from a URL asynchronously."""
        try:
            async with self.semaphore:
                async with session.get(url, headers=self.headers, timeout=10) as response:
                    if response.status == 200:
                        html = await response.text()
                        return self._clean_html(html)
                    return None
        except Exception:
            return None

    def _clean_html(self, html_content: str) -> str:
        """Strip HTML to raw text to save tokens."""
        soup = BeautifulSoup(html_content, 'html.parser')
        # Remove junk elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'iframe', 'svg']):
            element.decompose()
        # Get text and clean whitespace
        text = soup.get_text(separator=' ', strip=True)
        return ' '.join(text.split())[:5000]  # Limit length

class ClimateService:
    """Service for fetching climate data from Open-Meteo."""

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_climate_data(lat: float, lon: float) -> Optional[Dict]:
        """Fetch climate data - Wrapped in st.cache_data for performance."""
        # Using the synchronous library here as it's a single fast call and the library is sync
        # We could make it async but the library is optimized for dataframes
        cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "daily": ["temperature_2m_max", "temperature_2m_min", "rain_sum", "et0_fao_evapotranspiration", "precipitation_hours"],
            "timezone": "auto"
        }
        
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            
            daily = response.Daily()
            daily_temp_max = daily.Variables(0).ValuesAsNumpy()
            daily_temp_min = daily.Variables(1).ValuesAsNumpy()
            daily_rain_sum = daily.Variables(2).ValuesAsNumpy()
            daily_et0 = daily.Variables(3).ValuesAsNumpy()
            daily_precip_hours = daily.Variables(4).ValuesAsNumpy()
            
            avg_temp_max = float(daily_temp_max.mean())
            avg_temp_min = float(daily_temp_min.mean())
            avg_temp = (avg_temp_max + avg_temp_min) / 2
            total_rain = float(daily_rain_sum.sum())
            total_et0 = float(daily_et0.sum())
            frost_days = int((daily_temp_min < 0).sum())
            total_precip_hours = float(daily_precip_hours.sum())
            
            # Return as dict for simple serialization, or could return Model
            return ClimateData(
                average_temperature_c=round(avg_temp, 1),
                avg_temp_max_c=round(avg_temp_max, 1),
                avg_temp_min_c=round(avg_temp_min, 1),
                total_annual_rainfall_mm=round(total_rain, 1),
                total_annual_et0_mm=round(total_et0, 1),
                frost_days=frost_days,
                precipitation_hours=round(total_precip_hours, 1),
                water_balance=round(total_rain - total_et0, 1),
                climate_summary=f"Mean Temp: {avg_temp:.1f}°C, Annual Rain: {total_rain:.1f}mm, ET0: {total_et0:.1f}mm"
            ).model_dump()
            
        except Exception as e:
            print(f"Climate API Error: {e}")
            return None

class ListingService:
    """Handles searching for listings and coordinating the scraping."""
    
    def __init__(self, serper_api_key: str):
        self.api_key = serper_api_key

    @staticmethod
    def is_valid_listing_url(url: str) -> bool:
        """
        The 'Bouncer': Strict logic to reject index/search pages.
        Only allow actual property listing pages.
        """
        url_lower = url.lower()
        
        # 1. Immediate Disqualifiers (Blocklist)
        # Expanded based on user request to block aggregation pages
        blocklist = [
            "/label/", "/search", "/agencies/", "/agency/", "/town/", "/region/",
            "?ac=", "page=", "/sold/", "/auctions/", "/news/", "/blog/", 
            "/guide/", "/about/", "/contact/", "/team/", "/under-offer/",
            "realestate.com.au/buy/", # Generic buy pages
            "domain.com.au/sale/",    # Generic sale pages
        ]
        
        if any(b in url_lower for b in blocklist):
            return False
            
        # 2. Positive Signal Checks (Must look like a listing)
        # Farmbuy: usually ends in -123456
        if "farmbuy.com" in url_lower:
            return bool(re.search(r'-\d+$', url_lower)) or bool(re.search(r'-\d+/$', url_lower))
            
        # Generic: Check for ID patterns
        has_id_end = bool(re.search(r'-\d+/?$', url_lower))
        has_id_path = bool(re.search(r'/\d{6,}', url_lower)) # Long numeric ID
        has_property_id = "/property-" in url_lower or "/property/" in url_lower
        has_address_slug = bool(re.search(r'/\d+-[a-z]', url_lower)) # e.g. /123-smith-road
        
        return (has_id_end or has_id_path or has_property_id or has_address_slug)

    def search_listings(self, location_name: str) -> List[Dict]:
        """Search Serper for candidate URLs with strict filtering."""
        url = "https://google.serper.dev/search"
        headers = {'X-API-KEY': self.api_key, 'Content-Type': 'application/json'}
        
        # Exclusions to prevent index pages and sold properties
        exclusions = '-inurl:agencies -inurl:label -inurl:search -inurl:page -inurl:town -inurl:region -sold -under-offer'
        
        sites = [
            ("Farmbuy", f'site:farmbuy.com {location_name} "price" "acres" {exclusions}'),
            ("Elders", f'site:eldersrealestate.com.au {location_name} rural "price" {exclusions}'),
            ("RealEstate.com.au", f'site:realestate.com.au {location_name} rural land "price" {exclusions}'),
            ("Domain", f'site:domain.com.au {location_name} rural "price" {exclusions}'),
        ]
        
        candidates = []
        seen_urls = set()
        
        for _, query in sites:
            payload = json.dumps({"q": query, "num": 25})
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=10)
                data = response.json()
                if 'organic' in data:
                    for item in data['organic']:
                        item_url = item.get('link', '')
                        
                        # Dedup
                        if item_url in seen_urls:
                            continue
                            
                        # The Bouncer Check
                        if self.is_valid_listing_url(item_url):
                            seen_urls.add(item_url)
                            candidates.append(item)
            except Exception:
                pass
                    
        # If strict search yields no results, try fallback with less specific queries
        if not candidates:
            if "Region near" in location_name:
                return []
                
            for _, query in sites:
                # Remove some exclusions but keep the critical URL blockers
                fallback_query = query.replace('"price" "acres"', 'land for sale')
                payload = json.dumps({"q": fallback_query, "num": 20})
                
                try:
                    response = requests.post(url, headers=headers, data=payload, timeout=10)
                    data = response.json()
                    
                    if 'organic' in data:
                        for item in data['organic']:
                            item_url = item.get('link', '')
                            if item_url not in seen_urls:
                                if self.is_valid_listing_url(item_url):
                                    seen_urls.add(item_url)
                                    candidates.append(item)
                except Exception:
                    pass
                    
        return candidates

    async def get_listings_with_content(self, location_name: str, status_callback=None) -> List[ListingItem]:
        """
        Main async method: Search -> Filter -> Scrape concurrently.
        """
        # 1. Search (Blocking I/O but fast enough in thread)
        if status_callback: status_callback("Finding properties...", 0.1)
        # Fix for Python 3.8: Use run_in_executor instead of to_thread
        loop = asyncio.get_running_loop()
        raw_items = await loop.run_in_executor(None, self.search_listings, location_name)
        
        if not raw_items:
            return []

        # 2. Async Scrape
        if status_callback: status_callback(f"Scraping {len(raw_items)} sites in parallel...", 0.3)
        
        scraper = AsyncScraper()
        results = []
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for item in raw_items:
                tasks.append(scraper.fetch_text(session, item['link']))
            
            # Run all requests
            scraped_texts = await asyncio.gather(*tasks)
            
            # Merge results
            for item, text in zip(raw_items, scraped_texts):
                content = text if text else item.get('snippet', '')
                results.append(ListingItem(
                    title=item.get('title', 'Unknown'),
                    url=item.get('link'),
                    image=item.get('imageUrl') or item.get('thumbnail'),
                    scraped_content=content
                ))
        
        return results

class AnalysisEngine:
    """Handles Gemini AI generation via REST API (Python 3.8 safe)."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Updated to gemini-2.5-flash as requested by user
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={self.api_key}"
        self.headers = {'Content-Type': 'application/json'}

    def analyze(self, climate: Dict, listings: List[ListingItem], image_url: Optional[str] = None) -> AnalysisResponse:
        """
        Constructs prompt and gets analysis from Gemini via REST.
        """
        # Prepare small JSON for prompt
        listings_min = [
            {"title": l.title, "url": l.url, "content": l.scraped_content[:2000]} 
            for l in listings[:25]
        ]
        
        climate_summary = (
            f"Temp: {climate.get('average_temperature_c')}C, "
            f"Rain: {climate.get('total_annual_rainfall_mm')}mm, "
            f"ET0: {climate.get('total_annual_et0_mm')}mm, "
            f"Frost Days: {climate.get('frost_days')}"
        )

        prompt_text = f"""
        ACT AS: Expert Agricultural Investment Analyst. 
        TASK: Rank these property listings based on the provided CLIMATE DATA.
        
        CLIMATE DATA:
        {climate_summary}
        
        LISTINGS DATA ({len(listings_min)} items):
        {json.dumps(listings_min)}
        
        INSTRUCTIONS:
        1. FILTER: Ignore guide, blog, or aggregate pages. Keep only SPECIFIC properties for sale.
        2. EXCLUDE: Discard any property marked as SOLD, UNDER OFFER, or WITHDRAWN.
        3. EXTRACT: Price, Size (Acres/Ha).
        3. RANK: Score 0-100 based on data completeness + investment suitability given the climate (e.g. high rain = good).
        4. SUMMARY: Write a 'location_summary' and 'investor_summary' using the climate data.
        
        OUTPUT SCHEMA:
        Return valid JSON matching this structure exactly (NO MARKDOWN):
        {{
            "location_summary": "...",
            "suitability_score": 0-100,
            "water_security": "High" or "Needs Irrigation",
            "operation_difficulty": "Easy" or "Hard",
            "crop_versatility": "High" or "Low",
            "investor_summary": "...",
            "total_candidates_reviewed": int,
            "valid_listings_found": int,
            "listings_analysis": [
                {{
                    "title": "...",
                    "price": "...",
                    "size": "...",
                    "url": "...",
                    "relevance_score": int,
                    "investment_strategy": "..."
                }}
            ]
        }}
        """
        
        try:
            # Construct JSON Payload
            parts = [{"text": prompt_text}]
            
            # Handle image if present (Fetch and convert to base64 not needed for URL if we can't send it easily via REST without blob)
            # For simplicity in REST without complicated multipart, we might skip the image or use the image URL if supported.
            # Gemini Vision via REST usually expects base64 inline data.
            # Let's try to fetch and base64 encode it if we want to be fancy, OR just skip image for now to ensure stability.
            # Skipping image for stability on this fix. 
            
            payload = {
                "contents": [{
                    "parts": parts
                }],
                "generation_config": {
                    "response_mime_type": "application/json"
                }
            }
            
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            
            if response.status_code != 200:
                return AnalysisResponse(
                    location_summary="API Error", suitability_score=0, water_security="Error",
                    operation_difficulty="Error", crop_versatility="Error", investor_summary="Error",
                    total_candidates_reviewed=0, valid_listings_found=0, listings_analysis=[],
                    error=True, message=f"Gemini API Error {response.status_code}: {response.text}"
                )
                
            result = response.json()
            # Parse candidate text
            try:
                raw_text = result['candidates'][0]['content']['parts'][0]['text']
                return self._parse_json(raw_text)
            except (KeyError, IndexError):
                return AnalysisResponse(
                    location_summary="format Error", suitability_score=0, water_security="Error",
                    operation_difficulty="Error", crop_versatility="Error", investor_summary="Error",
                    total_candidates_reviewed=0, valid_listings_found=0, listings_analysis=[],
                    error=True, message="Unexpected API response format", raw_response=str(result)
                )
            
        except Exception as e:
            return AnalysisResponse(
                location_summary="Error", suitability_score=0, water_security="Error",
                operation_difficulty="Error", crop_versatility="Error", investor_summary="Error",
                total_candidates_reviewed=0, valid_listings_found=0, listings_analysis=[],
                error=True, message=str(e)
            )

    def _parse_json(self, text: str) -> AnalysisResponse:
        """Clean and parse JSON from LLM response."""
        try:
            # Strip markdown code blocks
            clean = re.sub(r'```(?:json)?', '', text).strip()
            # Clean possible trailing backticks
            clean = clean.strip('`')
            data = json.loads(clean)
            return AnalysisResponse(**data)
        except Exception as e:
            return AnalysisResponse(
                location_summary="Error parsing", suitability_score=0, water_security="Error",
                operation_difficulty="Error", crop_versatility="Error", investor_summary="Error",
                total_candidates_reviewed=0, valid_listings_found=0, listings_analysis=[],
                error=True, message=f"JSON Parse Error: {e}", raw_response=text
            )

