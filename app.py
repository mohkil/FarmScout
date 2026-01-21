import streamlit as st
import asyncio
import pandas as pd
from streamlit_folium import st_folium
import folium

# Import strictly typed models and services
from services import ClimateService, ListingService, AnalysisEngine, GeocodingService
from models import AnalysisResponse

# --- Configuration ---
st.set_page_config(layout="wide", page_title="FarmScout Pro", page_icon="🌾")

# --- Constants ---
DEFAULT_LAT, DEFAULT_LON = -32.2569, 148.6011  # Dubbo, NSW

# --- Initialization ---
if "results" not in st.session_state:
    st.session_state.results = None
if "is_analyzing" not in st.session_state:
    st.session_state.is_analyzing = False

# --- Helper Functions (View Layer) ---
def get_satellite_image_url(lat, lon, api_key):
    """Construct Google Static Maps URL."""
    base_url = "https://maps.googleapis.com/maps/api/staticmap"
    params = f"?center={lat},{lon}&zoom=12&size=640x400&scale=2&maptype=satellite&key={api_key}"
    return base_url + params

async def run_analysis_pipeline(lat, lon, address, api_keys, status_container):
    """
    Orchestrator for the analysis pipeline.
    Runs async scraping and sync API calls.
    """
    status_container.update(label="🌤️ Fetching Climate Data...", state="running")
    
    # 1. Climate Data (Fast, Cached)
    climate_service = ClimateService()
    climate_data = climate_service.get_climate_data(lat, lon)
    
    if not climate_data:
        raise Exception("Failed to fetch climate data. Service may be down.")
    
    status_container.write(f"✅ Climate: {climate_data['climate_summary']}")
    
    # 2. Listings Search & Scrape (Async, I/O Heavy)
    status_container.update(label=f"🚜 Searching listings near {address}...", state="running")
    listing_service = ListingService(api_keys['SERPER'])
    
    # Callback to update UI from deep within service if needed
    def update_progress(msg, pct):
        status_container.write(f"⏳ {msg}")
        
    listings = await listing_service.get_listings_with_content(address, status_callback=update_progress)
    
    if not listings:
        status_container.warning("No listings found. Trying nearby region...")
        # Simple fallback logic could go here or be inside method
    
    status_container.write(f"✅ Scraped {len(listings)} potential properties")
    
    # 3. AI Analysis (CPU/API Bound)
    if listings:
        status_container.update(label="🧠 AI Analyst Processing...", state="running")
        sat_url = get_satellite_image_url(lat, lon, api_keys['MAPS'])
        
        analysis_engine = AnalysisEngine(api_keys['GEMINI'])
        # Run in thread if blocking, but here we just call directly
        analysis = analysis_engine.analyze(climate_data, listings, sat_url)
    else:
        # Fallback analysis if no listings
        analysis = AnalysisResponse(
            location_summary="No listings found.", suitability_score=0, 
            water_security="N/A", operation_difficulty="N/A", crop_versatility="N/A", 
            investor_summary="N/A", total_candidates_reviewed=0, valid_listings_found=0, 
            listings_analysis=[], error=True, message="No listings found to analyze."
        )

    return {
        "address": address,
        "climate": climate_data,
        "listings": listings,
        "analysis": analysis,
        "sat_url": get_satellite_image_url(lat, lon, api_keys['MAPS'])
    }

# --- Main UI ---
st.title("🚜 FarmScout")
st.caption("Agricultural Investment Intelligence")

# Check Secrets
try:
    API_KEYS = {
        "GEMINI": st.secrets["general"]["GEMINI_API_KEY"],
        "MAPS": st.secrets["general"]["MAPS_API_KEY"],
        "SERPER": st.secrets["general"]["SERPER_API_KEY"]
    }
except:
    st.error("Missing .streamlit/secrets.toml")
    st.stop()

# --- State Management ---
if "map_clicked" not in st.session_state:
    st.session_state.map_clicked = None
if "selected_address" not in st.session_state:
    st.session_state.selected_address = None

# --- Responsive Layout ---
# Use a container for the map and controls to ensure they stack well on mobile
main_container = st.container()

with main_container:
    # 1. Map Section
    m = folium.Map(location=[DEFAULT_LAT, DEFAULT_LON], zoom_start=10)
    
    # If we have a selected location, show a marker with popup
    if st.session_state.map_clicked:
        clat, clon = st.session_state.map_clicked
        folium.Marker(
            [clat, clon], 
            popup=folium.Popup(f"<b>Selected:</b><br>{st.session_state.selected_address}", max_width=300),
            tooltip="Selected Location"
        ).add_to(m)
    
    m.add_child(folium.LatLngPopup())
    
    # Render map
    map_data = st_folium(m, height=500, width=1400, use_container_width=True)

    # Handle Map Clicks
    if map_data and map_data.get("last_clicked"):
        clicked_lat = map_data["last_clicked"]["lat"]
        clicked_lon = map_data["last_clicked"]["lng"]
        
        # Only update if clicked location is different to prevent loops
        if st.session_state.map_clicked != (clicked_lat, clicked_lon):
            st.session_state.map_clicked = (clicked_lat, clicked_lon)
            
            # Resolve Address Immediately
            geocoder = GeocodingService()
            address = geocoder.get_location_name(clicked_lat, clicked_lon)
            st.session_state.selected_address = address
            st.rerun()

    # 2. Controls Section (Sticky/Floating effect on mobile via simple vertical stacking)
    if st.session_state.map_clicked and st.session_state.selected_address:
        st.markdown("---")
        
        # Responsive Columns for Controls
        # On mobile, these will stack. On desktop, they align.
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.info(f"📍 **Selected Target:** {st.session_state.selected_address}")
            st.caption(f"Coordinates: {st.session_state.map_clicked[0]:.4f}, {st.session_state.map_clicked[1]:.4f}")
            
        with c2:
            # The "Analyze" action
            if st.button("🚀 Analyze Location", type="primary", use_container_width=True):
                lat, lon = st.session_state.map_clicked
                addr = st.session_state.selected_address
                
                with st.status("🚀 Running Analysis Pipeline...", expanded=True) as status:
                    try:
                        results = asyncio.run(run_analysis_pipeline(
                            lat, lon, addr, API_KEYS, status
                        ))
                        st.session_state.results = results
                        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    except Exception as e:
                        status.update(label="❌ Analysis Failed", state="error")
                        st.error(f"Error: {e}")




# Results Dashboard
if st.session_state.results:
    res = st.session_state.results
    analysis = res['analysis']
    
    st.markdown("---")
    
    # Custom CSS for the Dashboard
    st.markdown("""
    <style>
    /* Dashboard Cards */
    .dash-card { border-radius: 10px; padding: 20px; color: white; margin-bottom: 20px; }
    .dash-blue { background-color: #2c5282; }
    .dash-green { background-color: #387c2b; }
    .dash-gold { background-color: #b08d00; }
    .dash-icon { font-size: 2em; display: block; margin-bottom: 10px; text-align: center; }
    .dash-label { font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; text-align: center; display: block; }
    .dash-val { font-size: 1.4em; font-weight: bold; text-align: center; display: block; }
    
    /* Listing Cards */
    .prop-card { 
        background-color: white; 
        border: 1px solid #e0e0e0; 
        border-radius: 8px; 
        padding: 20px; 
        margin-bottom: 15px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .prop-header { display: flex; justify-content: space-between; align-items: flex-start; }
    .prop-title { font-size: 1.1em; font-weight: bold; color: #333; }
    .relevance-badge { 
        background-color: #6366f1; color: white; 
        padding: 4px 10px; border-radius: 12px; 
        font-size: 0.8em; font-weight: bold;
    }
    .prop-meta { color: #666; font-size: 0.9em; margin-top: 5px; }
    .prop-price { color: #0288d1; font-weight: bold; font-size: 1em; display: block; margin-top: 5px;}
    .insight-box { 
        background-color: #f0f9ff; 
        border-left: 4px solid #0288d1; 
        padding: 10px; margin-top: 15px; 
        font-size: 0.9em; color: #444; 
    }
    .details-btn {
        display: block; width: 100%; text-align: center;
        padding: 10px; margin-top: 15px;
        background-color: white; border: 1px solid #ddd;
        border-radius: 6px; text-decoration: none; color: #555;
        font-size: 0.9em;
    }
    .details-btn:hover { background-color: #f5f5f5; }
    </style>
    """, unsafe_allow_html=True)
    
    if not analysis.error:
        # Row 1: Terrain & Suitability
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.subheader("🛰️ Terrain Overview")
            st.image(res['sat_url'], use_container_width=True)
            
        with c2:
            st.subheader("📊 Investment Suitability")
            score = analysis.suitability_score
            st.metric("Suitability Score", f"{score}/100")
            
            # Simple Progress Bar
            st.progress(score)
            
            if score >= 80:
                st.success("✅ Excellent investment potential")
            elif score >= 60:
                st.info("ℹ️ Moderate potential - requires due diligence")
            else:
                st.warning("⚠️ Low potential or high risk detected")
                
            st.divider()
            cc1, cc2 = st.columns(2)
            cc1.metric("Avg Temp", f"{res['climate'].get('average_temperature_c')}°C")
            cc2.metric("Annual Rain", f"{res['climate'].get('total_annual_rainfall_mm')}mm")

        st.markdown("---")
        
        # Row 2: Investor Dashboard (Colored Cards)
        st.subheader("📊 Investor Dashboard")
        d1, d2, d3 = st.columns(3)
        
        # Helper to render html card
        def dashboard_card(icon, label, value, css_class):
            return f"""
            <div class="dash-card {css_class}">
                <span class="dash-icon">{icon}</span>
                <span class="dash-label">{label}</span>
                <span class="dash-val">{value}</span>
            </div>
            """
        
        with d1:
            st.markdown(dashboard_card("💧", "Water Status", analysis.water_security, "dash-blue"), unsafe_allow_html=True)
        with d2:
            st.markdown(dashboard_card("🚜", "Operational Ease", analysis.operation_difficulty, "dash-green"), unsafe_allow_html=True)
        with d3:
            st.markdown(dashboard_card("🌱", "Land Potential", analysis.crop_versatility, "dash-gold"), unsafe_allow_html=True)

        st.markdown("### 💼 Investor Summary")
        st.info(analysis.investor_summary)
        
        st.markdown("### 📝 Location Analysis")
        st.write(analysis.location_summary)
        
        st.markdown("---")

        # Row 3: Listings
        st.subheader("🏡 Top Investment Properties")
        st.caption(f"Showing {len(analysis.listings_analysis)} properties ranked by investment potential")
        
        for i, item in enumerate(analysis.listings_analysis, 1):
            # HTML Card render
            html_card = f"""
            <div class="prop-card">
                <div class="prop-header">
                    <span class="prop-title">#{i} — {item.title}</span>
                    <span class="relevance-badge">Relevance: {item.relevance_score}/100</span>
                </div>
                <div class="prop-price">{item.price}</div>
                <div class="prop-meta">📏 {item.size}</div>
                <div class="insight-box">
                    💡 <b>Investment Potential:</b> {item.investment_strategy}
                </div>
                <a class="details-btn" href="{item.url}" target="_blank">🔗 View Property Details</a>
            </div>
            """
            st.markdown(html_card, unsafe_allow_html=True)
            
    else:
        st.error(f"Analysis Failed: {analysis.message}")

