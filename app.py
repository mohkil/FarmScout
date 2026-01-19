import streamlit as st
import folium
from streamlit_folium import st_folium
import google.generativeai as genai
import requests
import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import json
import re
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration ---
st.set_page_config(layout="wide", page_title="FarmScout", page_icon="🌾")

# --- Constants ---
# Only exclude very obvious non-listing content
EXCLUDED_TITLE_KEYWORDS = [
    "market report", "buyers guide", "buyer's guide", 
    "news article", "blog post"
]

# --- API Keys Management ---
try:
    GEMINI_API_KEY = st.secrets["general"]["GEMINI_API_KEY"]
    MAPS_API_KEY = st.secrets["general"]["MAPS_API_KEY"]
    SERPER_API_KEY = st.secrets["general"]["SERPER_API_KEY"]
    
    if "YOUR_GEMINI_KEY" in GEMINI_API_KEY:
        st.error("Please update .streamlit/secrets.toml with your actual API Keys.")
        st.stop()
        
except FileNotFoundError:
    st.error("Secrets file not found. Please create .streamlit/secrets.toml")
    st.stop()
except KeyError:
    st.error("Missing keys in .streamlit/secrets.toml. Please check the structure.")
    st.stop()


# --- Functions ---

def get_climate_data(lat, lon):
    """Fetch climate data from Open-Meteo with advanced agronomic variables."""
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

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
        
        return {
            "average_temperature_c": round(avg_temp, 1),
            "avg_temp_max_c": round(avg_temp_max, 1),
            "avg_temp_min_c": round(avg_temp_min, 1),
            "total_annual_rainfall_mm": round(total_rain, 1),
            "total_annual_et0_mm": round(total_et0, 1),
            "frost_days": frost_days,
            "precipitation_hours": round(total_precip_hours, 1),
            "water_balance": round(total_rain - total_et0, 1),
            "climate_summary": f"Mean Temp: {avg_temp:.1f}°C, Annual Rain: {total_rain:.1f}mm, ET0: {total_et0:.1f}mm"
        }
    except Exception as e:
        st.error(f"Error fetching climate data: {e}")
        return None

def get_address_from_coords(lat, lon):
    """Reverse geocode coordinates to get address using Nominatim.
    If no valid town/city name at exact location, searches for nearest populated place."""
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "json",
    }
    headers = {
        "User-Agent": "FarmScout_App/1.0"
    }
    
    try:
        response = requests.get(url, params=params, headers=headers)
        data = response.json()
        
        address = data.get("address", {})
        town = address.get("town") or address.get("city") or address.get("locality") or address.get("village") or address.get("hamlet")
        state = address.get("state")
        
        if town and state:
            return f"{town}, {state}"
        elif town:
            return town
        elif state:
            # No valid town/city - search for nearest populated place
            return find_nearest_town(lat, lon, state)
        else:
            return find_nearest_town(lat, lon, None)
            
    except Exception as e:
        return find_nearest_town(lat, lon, None)

def find_nearest_town(lat, lon, state):
    """Search for the nearest town/city to use for property searches."""
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        "User-Agent": "FarmScout_App/1.0"
    }
    
    # Search for nearby places in expanding radius
    search_radii = [0.1, 0.25, 0.5, 1.0]  # degrees (~11km, ~28km, ~55km, ~110km)
    
    for radius in search_radii:
        viewbox = f"{lon - radius},{lat + radius},{lon + radius},{lat - radius}"
        params = {
            "format": "json",
            "bounded": 1,
            "viewbox": viewbox,
            "limit": 5,
            "featuretype": "city,town,village"
        }
        
        # Add country/state filter if available
        if state:
            params["q"] = f"town in {state}"
        else:
            params["q"] = "town"
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            results = response.json()
            
            if results and len(results) > 0:
                # Find the first result with a proper name
                for result in results:
                    place_type = result.get("type", "").lower()
                    if place_type in ["city", "town", "village", "locality", "hamlet", "administrative"]:
                        name = result.get("display_name", "").split(",")[0]
                        if name and len(name) > 1:
                            if state:
                                return f"{name}, {state}"
                            return name
        except:
            continue
    
    # Fallback: return coordinates with state if available
    if state:
        return f"Region near {lat:.2f}, {lon:.2f} ({state})"
    return f"Region near {lat:.2f}, {lon:.2f}"

def get_satellite_image_url(lat, lon, api_key):
    """Construct Google Static Maps URL for ~10km view with scale."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = f"?center={lat},{lon}&zoom=12&size=640x400&scale=2&maptype=satellite&key={api_key}"
    return base_url + params

def scrape_listing_page(url):
    """Scrape a listing page and extract visible text."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            text = soup.get_text(separator=' ', strip=True)
            return text[:5000]
        else:
            return None
    except Exception:
        return None

def is_valid_listing_title(title):
    """Check if title is likely a real listing vs a guide/report."""
    title_lower = title.lower()
    for keyword in EXCLUDED_TITLE_KEYWORDS:
        if keyword in title_lower:
            return False
    return True

def scrape_with_metadata(item):
    """Scrape a single item and return with metadata. Used for parallel processing."""
    url = item.get('link', '#')
    snippet = item.get('snippet', 'No description available.')
    
    scraped_text = scrape_listing_page(url)
    content = scraped_text if scraped_text else snippet
    
    return {
        "title": item.get('title', 'Untitled Property'),
        "url": url,
        "image": item.get('imageUrl') or item.get('thumbnail'),
        "scraped_content": content[:3000]
    }

def is_valid_listing_url(url):
    """
    Aggressively filter out category and search result pages.
    
    The "Digit Rule": A valid listing URL MUST contain:
    - A listing ID pattern (hyphen followed by numbers, e.g., -400483)
    - OR a street number at the start of the slug (e.g., /235-clearview-road)
    
    Returns:
        bool: True if URL appears to be a specific property listing
    """
    url_lower = url.lower()
    
    # EXPANDED BLOCKLIST - Explicitly ban these patterns
    blocklist_patterns = [
        "/region/",
        "/town/",
        "/label/",
        "/agencies/",
        "/agency/",
        "/state/",
        "/search",
        "/browse/",
        "/category/",
        "/suburb/",
        "-rural-property-search",
        "-property-search",
        "-real-estate",
        "-for-sale",
        "?page=",
        "?ac=",
        "page=",
        "/sold/",
        "/auctions/",
        "/news/",
        "/blog/",
        "/guide/",
        "/about/",
        "/contact/",
        "/team/",
    ]
    
    # Check blocklist first - reject if any pattern matches
    for pattern in blocklist_patterns:
        if pattern in url_lower:
            return False
    
    # THE DIGIT RULE: URL must have a listing identifier
    # Pattern 1: Listing ID at end of URL (e.g., -400483 or -12345)
    has_listing_id_at_end = bool(re.search(r'-\d+/?$', url_lower))
    
    # Pattern 2: Listing ID in the URL path (e.g., /400483- or /-12345-)
    has_listing_id_in_path = bool(re.search(r'/\d+-', url_lower))
    
    # Pattern 3: Street number at start of slug (e.g., /235-clearview or /12-main-street)
    # This matches paths like /235-clearview-road-town
    has_street_number = bool(re.search(r'/\d{1,5}-[a-z]', url_lower))
    
    # Pattern 4: Property ID in path (common format: /property/12345)
    has_property_id = bool(re.search(r'/property/\d+', url_lower))
    
    # Pattern 5: Listing with lot number (e.g., /lot-5-some-road)
    has_lot_number = bool(re.search(r'/lot-\d+', url_lower))
    
    # URL must match at least ONE of these patterns to be considered valid
    is_valid = (
        has_listing_id_at_end or 
        has_listing_id_in_path or 
        has_street_number or 
        has_property_id or
        has_lot_number
    )
    
    return is_valid


def get_land_listings(location_name, api_key):
    """
    Search for listings with expanded results, aggressive URL filtering, and parallel scraping.
    
    Features:
    - Fetches up to 30 results per source ("Wide Net")
    - AGGRESSIVE URL FILTERING: Uses "Digit Rule" to filter out category pages
    - Pre-filters obvious non-listings (guides, reports, etc.)
    - Parallel scraping with ThreadPoolExecutor
    
    Returns:
        List[Dict] with keys: 'title', 'url', 'image', 'scraped_content'
    """
    url = "https://google.serper.dev/search"
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    
    # Refined queries to prioritize specific listings
    # Using "price" and inurl:- to encourage listing pages with IDs
    sites = [
        ("Farmbuy", f'site:farmbuy.com {location_name} "price" inurl:-'),
        ("Elders", f'site:eldersrealestate.com.au {location_name} rural "price"'),
        ("RealEstate.com.au", f'site:realestate.com.au {location_name} rural land "price"'),
    ]
    
    # Collect all candidate items from search
    candidates = []
    
    for source, query in sites:
        # Request more results per source
        payload = json.dumps({"q": query, "num": 30})
        try:
            response = requests.post(url, headers=headers, data=payload, timeout=15)
            data = response.json()
            
            if 'organic' in data:
                for item in data['organic']:
                    title = item.get('title', '')
                    item_url = item.get('link', '')
                    
                    # AGGRESSIVE FILTERING: Check both title AND URL
                    if is_valid_listing_title(title) and is_valid_listing_url(item_url):
                        candidates.append(item)
                        
        except Exception as e:
            # Silently continue - don't block entire search
            pass
    
    # If no candidates after strict filtering, retry with relaxed URL filter
    # but still apply blocklist
    if not candidates:
        for source, query in sites:
            # Use simpler query as fallback
            fallback_query = query.replace('"price" inurl:-', 'land for sale').replace('"price"', 'land')
            payload = json.dumps({"q": fallback_query, "num": 20})
            try:
                response = requests.post(url, headers=headers, data=payload, timeout=15)
                data = response.json()
                if 'organic' in data:
                    for item in data['organic'][:15]:
                        item_url = item.get('link', '')
                        # Still apply blocklist even in fallback
                        url_lower = item_url.lower()
                        is_blocked = any(p in url_lower for p in [
                            "/agencies/", "/agency/", "/state/", "/region/", 
                            "-property-search", "/search", "?page=", "?ac="
                        ])
                        if not is_blocked:
                            candidates.append(item)
            except:
                pass
    
    # Parallel scraping with ThreadPoolExecutor
    results = []
    
    if candidates:
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all scraping tasks
            future_to_item = {
                executor.submit(scrape_with_metadata, item): item 
                for item in candidates
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_item):
                try:
                    result = future.result(timeout=15)
                    if result:
                        results.append(result)
                except Exception:
                    # On scrape failure, still include with snippet only
                    item = future_to_item[future]
                    results.append({
                        "title": item.get('title', 'Untitled Property'),
                        "url": item.get('link', '#'),
                        "image": item.get('imageUrl') or item.get('thumbnail'),
                        "scraped_content": item.get('snippet', 'No description available.')
                    })
    
    return results

def parse_ai_response(response_text):
    """
    Parse AI response, handling potential markdown formatting.
    
    Returns:
        dict: Parsed JSON or error dict
    """
    text = response_text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r'^```(?:json)?\s*\n?', '', text)
        text = re.sub(r'\n?```\s*$', '', text)
    
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        return {
            "error": True,
            "message": f"Failed to parse AI response: {str(e)}",
            "raw_response": response_text[:500]
        }

def analyze_location(climate, listings, image_url, ai_key):
    """
    Analyze location using Gemini with ranking engine.
    
    Features:
    - Investment Filter persona
    - Identifies valid listings vs generic content
    - Assigns Relevance Score (0-100)
    - Returns Top 15 sorted by score
    
    Returns:
        dict: Structured analysis with location_summary, suitability_score, listings_analysis
    """
    try:
        genai.configure(api_key=ai_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare listing data for Gemini (send all candidates for ranking)
        listings_for_ai = []
        for l in listings[:25]:  # Send up to 25 for ranking
            listings_for_ai.append({
                "title": l['title'],
                "url": l['url'],
                "content": l['scraped_content'][:2000]
            })
        
        listings_json = json.dumps(listings_for_ai, indent=2)
        
        prompt = f"""You are an Investment Risk Analyst specializing in agricultural land. I will provide raw text from ~{len(listings_for_ai)} web pages and detailed climate data. Your job is to:

1. **IDENTIFY**: Is this a SPECIFIC property listing for sale? If it's a generic guide, blog post, market report, or aggregate page with multiple listings, IGNORE it completely.

2. **EXTRACT**: For each valid listing, extract:
   - Price (formatted like "$1,200,000" or "Contact Agent" if not found)
   - Size (formatted like "150 Acres" or "Not specified")
   - Title (from input)
   - URL (from input)

3. **EVALUATE**: Assign a "relevance_score" (0-100) based on:
   - Data completeness (price + size = higher score)
   - Investment quality signals (water access, improved pastures, infrastructure)
   - Clarity of listing information

4. **INVESTOR METRICS** (CRITICAL - Use climate data to determine these):
   - Compare 'rain_sum' vs 'et0_fao_evapotranspiration' to assess water security
   - Look at 'temperature_2m_min' and frost_days to estimate frost risk
   - Analyze the satellite terrain image for operational complexity

5. **CONSTRAINT**: If a listing lacks BOTH a Price AND a Size, do not include it unless the description is exceptionally detailed about the property.

6. **OUTPUT**: Return ONLY the Top 15 valid listings, sorted by relevance_score descending.

CRITICAL: Output ONLY valid JSON. No markdown, no code blocks, no backticks.

Schema:
{{
  "location_summary": "A professional paragraph describing the terrain, climate, and agricultural suitability. State facts directly.",
  "suitability_score": <integer 0-100>,
  "water_security": "High Security" if total rain > total ET0, otherwise "Needs Irrigation",
  "operation_difficulty": "Easy / Flat" if terrain appears flat/gentle, otherwise "Complex / Steep",
  "crop_versatility": "High Versatility" if climate supports many crops (low frost, good water), otherwise "Limited / Grazing Only",
  "investor_summary": "A 2-sentence plain English summary explaining water security, terrain difficulty, and crop potential for investors.",
  "total_candidates_reviewed": <number>,
  "valid_listings_found": <number>,
  "listings_analysis": [
    {{
      "title": "<title from input>",
      "price": "<extracted price>",
      "size": "<extracted size>",
      "url": "<url from input>",
      "relevance_score": <integer 0-100>,
      "investment_strategy": "<specific recommendation for THIS property>"
    }}
  ]
}}

INPUT DATA:

Climate:
- Avg Temp: {climate['average_temperature_c']}°C
- Annual Rainfall: {climate['total_annual_rainfall_mm']}mm
- Annual ET0 (Evapotranspiration): {climate.get('total_annual_et0_mm', 'N/A')}mm
- Water Balance (Rain - ET0): {climate.get('water_balance', 'N/A')}mm
- Frost Days: {climate.get('frost_days', 'N/A')} days
- Avg Min Temp: {climate.get('avg_temp_min_c', 'N/A')}°C

Candidates ({len(listings_for_ai)} pages):
{listings_json}

Output ONLY the JSON object."""

        # Try to include satellite image for terrain analysis
        image_parts = []
        if image_url:
            try:
                img_response = requests.get(image_url, timeout=10)
                if img_response.status_code == 200:
                    image_parts = [{"mime_type": "image/png", "data": img_response.content}]
            except:
                pass

        if image_parts:
            response = model.generate_content([prompt, image_parts[0]])
        else:
            response = model.generate_content(prompt)
        
        return parse_ai_response(response.text)
        
    except Exception as e:
        return {
            "error": True,
            "message": f"AI analysis failed: {str(e)}"
        }


# --- Main Interface ---

st.title("🚜 FarmScout")
st.caption("Agricultural Investment Intelligence Platform")

# Custom CSS for standardized cards
st.markdown("""
<style>
.property-card {
    padding: 1rem;
    margin-bottom: 0.5rem;
}
.property-card h4 {
    margin: 0 0 0.5rem 0;
    font-size: 1.1rem;
    line-height: 1.3;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}
.property-price {
    color: #1f77b4;
    font-size: 1.5rem;
    font-weight: bold;
    margin: 0.5rem 0;
}
.property-size {
    font-weight: bold;
    font-size: 1rem;
    margin: 0.25rem 0;
}
.property-score {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 1rem;
    font-size: 0.85rem;
    display: inline-block;
    margin: 0.25rem 0;
}
.property-strategy {
    background-color: #f0f7ff;
    border-left: 3px solid #1f77b4;
    padding: 0.75rem;
    margin: 0.75rem 0;
    font-size: 0.9rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# CSS to disable map during analysis
map_disabled_css = """
<style>
.map-disabled-overlay {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0, 0, 0, 0.4);
    z-index: 999999 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    pointer-events: all !important;
}
.map-disabled-overlay .message {
    background: white;
    padding: 2rem 3rem;
    border-radius: 12px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    text-align: center;
}
/* Also disable iframe interactions */
.stApp iframe {
    pointer-events: none !important;
    opacity: 0.5;
}
</style>
"""

# Default values
default_lat, default_lon = -32.2569, 148.6011  # Dubbo, NSW
# default_lat, default_lon = None, None

# Check if analysis is in progress
is_analyzing = st.session_state.get("is_analyzing", False)

# Helper for overlay HTML
def render_overlay(msg):
    return f"""
    <div class="map-disabled-overlay">
        <div class="message">
            <h3>⏳ Analysis in Progress</h3>
            <p>{msg}</p>
        </div>
    </div>
    """

# Placeholder for overlay - defined at top level to ensure it's always in the same spot
overlay_placeholder = st.empty()

# Show overlay and disable map if analysis is in progress
if is_analyzing:
    st.markdown(map_disabled_css, unsafe_allow_html=True)
    overlay_placeholder.markdown(render_overlay("Please wait while we analyze the location..."), unsafe_allow_html=True)

m = folium.Map(location=[default_lat, default_lon], zoom_start=10)
m.add_child(folium.LatLngPopup())

# Render map - clicks will be blocked by CSS overlay during analysis
map_data = st_folium(m, height=450, width=1400, use_container_width=True)

# Track if user has clicked on the map
clicked_lat = None
clicked_lon = None
location_selected = False

if map_data and map_data.get("last_clicked"):
    clicked_lat = map_data["last_clicked"]["lat"]
    clicked_lon = map_data["last_clicked"]["lng"]
    location_selected = True
else:
    # Use default for display only
    clicked_lat = default_lat
    clicked_lon = default_lon

# Get address immediately when location is clicked
current_address = None
if location_selected:
    norm_lon = (clicked_lon + 180) % 360 - 180
    current_address = get_address_from_coords(clicked_lat, norm_lon)

# Sidebar Controls
with st.sidebar:
    st.header("📍 Location")
    
    # Instruction text
    st.info("📍 Click a point on the map, then click 'Analyze Location' below.")
    
    # Show address if location selected
    if location_selected and current_address:
        st.success(f"**{current_address}**")
    elif not location_selected:
        st.warning("No location selected")
    
    # Coordinates - show "---" if no location selected
    if location_selected:
        st.metric("Latitude", f"{clicked_lat:.4f}")
        st.metric("Longitude", f"{clicked_lon:.4f}")
    else:
        st.metric("Latitude", "---")
        st.metric("Longitude", "---")
    
    st.divider()
    
    # Analyze button - disabled if no location selected OR analysis in progress
    analyze_btn = st.button(
        "🔍 Analyze Location" if not is_analyzing else "⏳ Analyzing...", 
        type="primary", 
        use_container_width=True,
        disabled=(not location_selected) or is_analyzing
    )

# Initialize Session State
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "results" not in st.session_state:
    st.session_state.results = {}
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False
if "analysis_coords" not in st.session_state:
    st.session_state.analysis_coords = None

# Analysis Control Flow
# 1. Trigger analysis when button clicked
if analyze_btn and location_selected:
    st.session_state.is_analyzing = True
    st.session_state.analysis_coords = {
        "lat": clicked_lat,
        "lon": clicked_lon,
        "address": current_address
    }
    st.rerun()

# 2. Run analysis if in analyzing state
if st.session_state.is_analyzing and st.session_state.analysis_coords:
    # Retrieve persisted coordinates
    a_lat = st.session_state.analysis_coords["lat"]
    a_lon = st.session_state.analysis_coords["lon"]
    a_addr = st.session_state.analysis_coords.get("address")
    
    norm_lon = (a_lon + 180) % 360 - 180
    
    # Initial "Start" message using the global placeholder
    overlay_placeholder.markdown(render_overlay("Initializing analysis engine..."), unsafe_allow_html=True)
    
    # Use st.status for smart loading experience (still useful for history/logs below overlay)
    with st.status("🔍 Analyzing location...", expanded=True) as status:
        try:
            # A. Geocoding
            overlay_placeholder.markdown(render_overlay("Resolving location details..."), unsafe_allow_html=True)
            status.write("📍 Resolving location name...")
            address = a_addr if a_addr else get_address_from_coords(a_lat, norm_lon)
            status.write(f"✅ Location: **{address}**")
            
            # B. Satellite Image
            overlay_placeholder.markdown(render_overlay("Fetching satellite imagery..."), unsafe_allow_html=True)
            status.write("🛰️ Fetching satellite imagery...")
            sat_url = get_satellite_image_url(a_lat, norm_lon, MAPS_API_KEY)
            status.write("✅ Satellite image ready")
            
            # C. Climate Data
            overlay_placeholder.markdown(render_overlay("Analyzing climate patterns..."), unsafe_allow_html=True)
            status.write("🌤️ Fetching climate data...")
            climate_data = get_climate_data(a_lat, norm_lon)
            if climate_data:
                status.write(f"✅ Climate: {climate_data['climate_summary']}")
            else:
                status.write("⚠️ Climate data unavailable")
            
            # D. Listings
            overlay_placeholder.markdown(render_overlay("Searching real estate markets..."), unsafe_allow_html=True)
            status.write("🚜 Searching real estate markets...")
            search_location = address
            listings = get_land_listings(search_location, SERPER_API_KEY)
            
            # If no listings found, try expanding
            if not listings:
                overlay_placeholder.markdown(render_overlay("Expanding search radius..."), unsafe_allow_html=True)
                status.write("⚠️ No listings at exact location. Searching nearby areas...")
                
                nearby_offsets = [
                    (0.2, 0), (-0.2, 0), (0, 0.2), (0, -0.2),
                    (0.5, 0), (-0.5, 0),
                ]
                
                for lon_offset, lat_offset in nearby_offsets:
                    nearby_lat = a_lat + lat_offset
                    nearby_lon = norm_lon + lon_offset
                    nearby_address = get_address_from_coords(nearby_lat, nearby_lon)
                    
                    if nearby_address == address or nearby_address.startswith("Region near"):
                        continue
                    
                    status.write(f"🔍 Trying: {nearby_address}...")
                    listings = get_land_listings(nearby_address, SERPER_API_KEY)
                    
                    if listings:
                        search_location = nearby_address
                        status.write(f"✅ Found listings near: **{nearby_address}**")
                        break
            
            status.write(f"✅ Found {len(listings)} potential listings")
            
            if listings:
                overlay_placeholder.markdown(render_overlay(f"Scraping {len(listings)} listings..."), unsafe_allow_html=True)
                status.write(f"🕷️ Scraping {len(listings)} listings in parallel...")
                status.write("⏱️ This may take 15-30 seconds...")
            
            # E. AI Analysis
            if climate_data and listings:
                overlay_placeholder.markdown(render_overlay("AI is evaluating investment potential..."), unsafe_allow_html=True)
                status.write("🧠 AI is ranking top investment opportunities...")
                analysis = analyze_location(climate_data, listings, sat_url, GEMINI_API_KEY)
                if not analysis.get("error"):
                    valid_count = analysis.get("valid_listings_found", len(analysis.get("listings_analysis", [])))
                    status.write(f"✅ Identified {valid_count} investment-grade properties")
            elif climate_data:
                analysis = {"error": True, "message": "No listings found for this location."}
            else:
                analysis = {"error": True, "message": "Could not fetch climate data."}
            
            # Store results
            st.session_state.results = {
                "address": address,
                "sat_url": sat_url,
                "climate_data": climate_data,
                "listings": listings,
                "analysis": analysis
            }
            st.session_state.analysis_done = True
            
            # Update status and reset analyzing flag
            st.session_state.is_analyzing = False
            
            if not analysis.get("error"):
                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                st.rerun()
            else:
                status.update(label="⚠️ Analysis completed with issues", state="error", expanded=True)
                st.rerun()
            
        except Exception as e:
            st.session_state.is_analyzing = False
            status.update(label="❌ Analysis Failed", state="error")
            st.error(f"An error occurred during analysis: {e}")
            # Ensure overlay is cleared on error (though st.error usually prints below)
            # Maybe keep overlay but show error? 
            # For now, let the error persist in the UI below, but likely the overlay blocks it.
            # Let's update the overlay to show error too!
            overlay_placeholder.markdown(f"""
            <div class="map-disabled-overlay">
                <div class="message">
                    <h3>❌ Analysis Failed</h3>
                    <p>{str(e)}</p>
                    <p>Refer to the sidebar or refresh the page.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Results Display
if st.session_state.analysis_done:
    res = st.session_state.results
    analysis = res.get("analysis", {})
    
    # Header with region
    if "address" in res:
        st.success(f"**📍 {res['address']}**")

    st.divider()
    
    # ============================================
    # SECTION 1: LOCATION ANALYSIS
    # ============================================
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("🛰️ Terrain Overview")
        if "sat_url" in res:
            st.image(res["sat_url"], use_container_width=True)
    
    with col_right:
        st.subheader("📊 Investment Suitability")
        
        # Suitability Score
        if not analysis.get("error") and "suitability_score" in analysis:
            score = analysis["suitability_score"]
            st.metric("Suitability Score", f"{score}/100")
            st.progress(score / 100)
            
            # Score interpretation
            if score >= 80:
                st.success("Excellent investment potential")
            elif score >= 60:
                st.info("Good investment opportunity")
            elif score >= 40:
                st.warning("Moderate potential - due diligence advised")
            else:
                st.error("Limited agricultural suitability")
        
        # Climate metrics
        if res.get("climate_data"):
            st.divider()
            c_data = res["climate_data"]
            col_t, col_r = st.columns(2)
            col_t.metric("🌡️ Avg Temp", f"{c_data['average_temperature_c']}°C")
            col_r.metric("🌧️ Annual Rain", f"{c_data['total_annual_rainfall_mm']}mm")
        
        # Ranking stats
        if not analysis.get("error"):
            st.divider()
            reviewed = analysis.get("total_candidates_reviewed", len(res.get("listings", [])))
            valid = analysis.get("valid_listings_found", len(analysis.get("listings_analysis", [])))
            st.caption(f"📊 Reviewed {reviewed} pages → {valid} valid listings")
    
    # ============================================
    # INVESTOR DASHBOARD - NEW SECTION
    # ============================================
    if not analysis.get("error"):
        st.divider()
        st.markdown("### 📊 Investor Dashboard")
        
        # Three-column metric display
        dash_col1, dash_col2, dash_col3 = st.columns(3)
        
        with dash_col1:
            water_security = analysis.get("water_security", "Unknown")
            water_icon = "✅" if "High" in water_security else "⚠️"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">💧</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Water Status</div>
                <div style="font-size: 1.3rem; font-weight: bold; margin-top: 0.5rem;">{water_icon} {water_security}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with dash_col2:
            operation_diff = analysis.get("operation_difficulty", "Unknown")
            op_icon = "✅" if "Easy" in operation_diff else "⚠️"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #2d5016 0%, #4a7c23 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚜</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Operational Ease</div>
                <div style="font-size: 1.3rem; font-weight: bold; margin-top: 0.5rem;">{op_icon} {operation_diff}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with dash_col3:
            crop_vers = analysis.get("crop_versatility", "Unknown")
            crop_icon = "✅" if "High" in crop_vers else "⚠️"
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #7c4a03 0%, #b8860b 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 2rem; margin-bottom: 0.5rem;">🌱</div>
                <div style="font-size: 0.9rem; opacity: 0.9;">Land Potential</div>
                <div style="font-size: 1.3rem; font-weight: bold; margin-top: 0.5rem;">{crop_icon} {crop_vers}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Investor Summary
        investor_summary = analysis.get("investor_summary", "")
        if investor_summary:
            st.markdown(f"""
            <div style="background-color: #f8f9fa; border-left: 4px solid #667eea; 
                        padding: 1rem 1.5rem; margin-top: 1rem; border-radius: 0 8px 8px 0;">
                <strong>💼 Investor Summary:</strong> {investor_summary}
            </div>
            """, unsafe_allow_html=True)
    
    # Location Summary
    if not analysis.get("error") and "location_summary" in analysis:
        st.divider()
        st.markdown("### 📋 Location Analysis")
        st.markdown(analysis["location_summary"])
    
    # Error handling for analysis
    if analysis.get("error"):
        st.divider()
        st.error(f"⚠️ {analysis.get('message', 'Analysis could not be completed.')}")
        if "raw_response" in analysis:
            with st.expander("View raw response"):
                st.code(analysis["raw_response"])
    
    st.divider()
    
    # ============================================
    # SECTION 2: ACTIVE LISTINGS (Standardized Cards)
    # ============================================
    
    st.subheader("🏡 Top Investment Properties")
    
    listings_analysis = analysis.get("listings_analysis", []) if not analysis.get("error") else []
    
    # Limit to top 15
    listings_to_show = listings_analysis[:15]
    
    if listings_to_show:
        st.caption(f"Showing {len(listings_to_show)} properties ranked by investment potential")
        
        # Display in vertical stack with standardized HTML cards
        for idx, listing in enumerate(listings_to_show, 1):
            with st.container(border=True):
                # Rank badge and title
                title = listing.get('title', 'Property')
                price = listing.get('price', 'Contact Agent')
                size = listing.get('size', 'Not specified')
                rel_score = listing.get('relevance_score', 0)
                strategy = listing.get('investment_strategy', '')
                url = listing.get('url', '#')
                
                # Custom HTML for consistent styling
                st.markdown(f"""
                <div class="property-card">
                    <h4>#{idx} — {title}</h4>
                    <p class="property-price">{price}</p>
                    <p class="property-size">📏 {size}</p>
                    <span class="property-score">⭐ Relevance: {rel_score}/100</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Investment Strategy in styled box
                if strategy:
                    st.markdown(f"""
                    <div class="property-strategy">
                        💡 <strong>Investment Potential:</strong> {strategy}
                    </div>
                    """, unsafe_allow_html=True)
                
                # Action button (Streamlit native for functionality)
                if url and url != '#':
                    st.link_button("🔗 View Property Details", url, use_container_width=True)
    else:
        # Fallback to raw listings if AI analysis failed
        raw_listings = res.get("listings", [])
        if raw_listings:
            st.warning("Displaying raw listing data (AI ranking unavailable)")
            for listing in raw_listings[:15]:
                with st.container(border=True):
                    st.markdown(f"**{listing.get('title', 'Property')}**")
                    if listing.get('image'):
                        st.image(listing['image'], width=300)
                    if listing.get('url'):
                        st.link_button("View Details", listing['url'], use_container_width=True)
        else:
            st.info("No properties found for this location. Try selecting a different region with more agricultural activity.")

