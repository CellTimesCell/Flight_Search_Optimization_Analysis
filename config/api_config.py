import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional


class APIConfig:
    """Configuration settings for external APIs"""

    # Amadeus API Configuration
    AMADEUS_API_KEY = "your_api_key"
    AMADEUS_API_SECRET = "your_api_secret"
    AMADEUS_BASE_URL = "https://test.api.amadeus.com"
    AMADEUS_TOKEN_URL = f"{AMADEUS_BASE_URL}/v1/security/oauth2/token"
    AMADEUS_FLIGHT_SEARCH_URL = f"{AMADEUS_BASE_URL}/v2/shopping/flight-offers"
    AMADEUS_REQUEST_LIMIT = 400

    # AviationStack API Configuration
    AVIATIONSTACK_API_KEY = "your_api"
    AVIATIONSTACK_BASE_URL = "http://api.aviationstack.com/v1"
    AVIATIONSTACK_REQUEST_LIMIT = 100

    # Database Configuration
    DATABASE_PATH = "data/flights.db"

    # Project Paths
    DATA_RAW_PATH = "data/raw"
    DATA_PROCESSED_PATH = "data/processed"
    DATA_EXTERNAL_PATH = "data/external"

    @classmethod
    def get_target_routes(cls) -> List[Dict[str, str]]:
        """Returns list of target routes for data collection"""
        return [
            {"origin": "NYC", "destination": "LON", "priority": 1, "category": "international"},
            {"origin": "NYC", "destination": "PAR", "priority": 1, "category": "international"},
            {"origin": "LAX", "destination": "NRT", "priority": 1, "category": "international"},
            {"origin": "NYC", "destination": "LAX", "priority": 1, "category": "domestic"},
            {"origin": "LON", "destination": "FRA", "priority": 2, "category": "european"},
            {"origin": "PAR", "destination": "ROM", "priority": 2, "category": "european"},
            {"origin": "NYC", "destination": "CHI", "priority": 2, "category": "domestic"},
            {"origin": "NYC", "destination": "BCN", "priority": 3, "category": "tourism"},
        ]

    @classmethod
    def get_target_dates(cls) -> List[str]:
        """Returns list of target dates for flight search"""
        dates = []
        base_date = datetime.now() + timedelta(days=14)

        # Weekly dates for next 2 months
        for i in range(8):
            date = base_date + timedelta(days=i * 7)
            dates.append(date.strftime("%Y-%m-%d"))

        # Monthly dates for next 3 months
        for i in range(3):
            date = base_date + timedelta(days=60 + i * 30)
            dates.append(date.strftime("%Y-%m-%d"))

        return dates
