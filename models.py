from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Union

class ClimateData(BaseModel):
    """Structured climate data from Open-Meteo."""
    average_temperature_c: float
    avg_temp_max_c: float
    avg_temp_min_c: float
    total_annual_rainfall_mm: float
    total_annual_et0_mm: float
    frost_days: int
    precipitation_hours: float
    water_balance: float
    climate_summary: str

class ListingItem(BaseModel):
    """A raw listing found via search."""
    title: str
    url: str
    image: Optional[str] = None
    scraped_content: str = Field(default="", description="The raw text content scraped from the page")

class AnalyzedListing(BaseModel):
    """A single listing that has been analyzed and extracted by AI."""
    title: str
    price: str
    size: str
    url: str
    relevance_score: int
    investment_strategy: str

class AnalysisResponse(BaseModel):
    """The complete structured response from the Gemini Analysis Engine."""
    location_summary: str
    suitability_score: int
    water_security: str
    operation_difficulty: str
    crop_versatility: str
    investor_summary: str
    total_candidates_reviewed: int
    valid_listings_found: int
    listings_analysis: List[AnalyzedListing]
    error: bool = False
    message: Optional[str] = None
    raw_response: Optional[str] = None
