import requests
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json


class AmadeusAPI:
    """Client for Amadeus Flight API with proper authentication"""

    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.token_url = f"{base_url}/v1/security/oauth2/token"
        self.flight_search_url = f"{base_url}/v2/shopping/flight-offers"

        self.access_token = None
        self.token_expires_at = None
        self.requests_made = 0
        self.max_requests = 400
        self.api_working = True

    def test_api_connection(self) -> bool:
        """Test if API credentials are working"""
        try:
            token = self.get_access_token()
            if token:
                print("SUCCESS: Amadeus API connection established")
                return True
            else:
                print("FAILED: Amadeus API authentication failed")
                return False
        except Exception as e:
            print(f"ERROR: Amadeus API connection test failed: {e}")
            return False

    def get_access_token(self) -> Optional[str]:
        """Get or refresh access token using correct credentials"""
        if self.access_token and self.token_expires_at > datetime.now():
            return self.access_token

        headers = {'Content-Type': 'application/x-www-form-urlencoded'}

        # Use the correct client_id and client_secret
        data = {
            'grant_type': 'client_credentials',
            'client_id': self.api_key,
            'client_secret': self.api_secret
        }

        try:
            print("Requesting Amadeus access token...")
            response = requests.post(self.token_url, headers=headers, data=data, timeout=30)

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data['access_token']
                expires_in = token_data.get('expires_in', 1799)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in)

                print(f"SUCCESS: Amadeus token obtained, expires in {expires_in} seconds")
                self.api_working = True
                return self.access_token

            elif response.status_code == 401:
                print(f"ERROR: 401 Unauthorized - Check API credentials")
                print(f"API Key: {self.api_key[:10]}...")
                print(f"API Secret: {self.api_secret[:10]}...")
                print(f"Response: {response.text}")
                self.api_working = False
                return None

            else:
                print(f"ERROR: HTTP {response.status_code}: {response.text}")
                self.api_working = False
                return None

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Network error during authentication: {e}")
            self.api_working = False
            return None

    def search_flights(self, origin: str, destination: str, departure_date: str,
                       return_date: Optional[str] = None, adults: int = 1,
                       max_results: int = 20) -> Dict:
        """Search for flights using Amadeus API"""

        if not self.api_working:
            print("API not working, skipping request")
            return {}

        if self.requests_made >= self.max_requests:
            print(f"Amadeus API request limit reached: {self.max_requests}")
            return {}

        token = self.get_access_token()
        if not token:
            print("No valid token available")
            return {}

        headers = {'Authorization': f'Bearer {token}'}
        params = {
            'originLocationCode': origin,
            'destinationLocationCode': destination,
            'departureDate': departure_date,
            'adults': adults,
            'max': max_results,
            'currencyCode': 'USD'
        }

        if return_date:
            params['returnDate'] = return_date

        try:
            print(f"Searching flights: {origin} -> {destination} on {departure_date}")
            response = requests.get(
                self.flight_search_url,
                headers=headers,
                params=params,
                timeout=30
            )

            self.requests_made += 1

            if response.status_code == 200:
                data = response.json()
                flight_count = len(data.get('data', []))
                print(f"SUCCESS: Found {flight_count} flights ({self.requests_made}/{self.max_requests} requests used)")
                return data

            elif response.status_code == 400:
                print(f"ERROR: Bad request - {response.text}")
                if "INVALID" in response.text.upper():
                    print(f"Note: {origin} or {destination} might be invalid airport codes")
                return {}

            else:
                print(f"ERROR: Flight search failed with status {response.status_code}")
                print(f"Response: {response.text}")
                return {}

        except requests.exceptions.RequestException as e:
            print(f"ERROR: Network error during flight search: {e}")
            return {}

    def parse_flight_offers(self, api_response: Dict, search_route: Dict,
                            search_date: str) -> List[Dict]:
        """Parse Amadeus API response into structured flight data"""

        flights = []

        if 'data' not in api_response:
            return flights

        # Extract dictionaries for reference
        dictionaries = api_response.get('dictionaries', {})
        carriers = dictionaries.get('carriers', {})
        aircraft = dictionaries.get('aircraft', {})

        print(f"Parsing {len(api_response['data'])} flight offers...")

        for offer in api_response['data']:
            try:
                # Get price information
                price_info = offer.get('price', {})
                total_price = float(price_info.get('total', 0))
                currency = price_info.get('currency', 'USD')

                # Process each itinerary
                for itinerary in offer.get('itineraries', []):
                    # Calculate total duration and stops
                    segments = itinerary.get('segments', [])
                    total_duration = self._parse_duration(itinerary.get('duration'))
                    stops = len(segments) - 1

                    # Process each segment
                    for segment_idx, segment in enumerate(segments):
                        departure = segment.get('departure', {})
                        arrival = segment.get('arrival', {})

                        # Extract airline information
                        carrier_code = segment.get('carrierCode')
                        airline_name = carriers.get(carrier_code, carrier_code)

                        # Extract aircraft information
                        aircraft_code = segment.get('aircraft', {}).get('code')
                        aircraft_type = aircraft.get(aircraft_code, aircraft_code)

                        flight_data = {
                            'origin': departure.get('iataCode'),
                            'destination': arrival.get('iataCode'),
                            'departure_date': search_date,
                            'departure_time': self._extract_time(departure.get('at')),
                            'arrival_time': self._extract_time(arrival.get('at')),
                            'airline_code': carrier_code,
                            'airline_name': airline_name,
                            'flight_number': f"{carrier_code}{segment.get('number', '')}",
                            'aircraft_type': aircraft_type,
                            'duration_minutes': total_duration,
                            'stops': stops,
                            'price_usd': total_price,
                            'currency': currency,
                            'booking_class': segment.get('cabin'),
                            'seats_available': segment.get('numberOfBookableSeats'),
                            'raw_data': offer
                        }

                        flights.append(flight_data)

            except (KeyError, ValueError, TypeError) as e:
                print(f"Error parsing flight offer: {e}")
                continue

        print(f"Successfully parsed {len(flights)} flight records")
        return flights

    def _parse_duration(self, duration_str: str) -> Optional[int]:
        """Parse ISO 8601 duration string to minutes"""
        if not duration_str:
            return None

        # Parse PT2H30M format
        hours_match = re.search(r'(\d+)H', duration_str)
        minutes_match = re.search(r'(\d+)M', duration_str)

        total_minutes = 0
        if hours_match:
            total_minutes += int(hours_match.group(1)) * 60
        if minutes_match:
            total_minutes += int(minutes_match.group(1))

        return total_minutes if total_minutes > 0 else None

    def _extract_time(self, datetime_str: str) -> Optional[str]:
        """Extract time from ISO datetime string"""
        if not datetime_str:
            return None

        try:
            # Parse ISO format: 2024-07-15T14:30:00
            dt = datetime.fromisoformat(datetime_str.replace('Z', '+00:00'))
            return dt.strftime('%H:%M')
        except (ValueError, AttributeError):
            return None

    def get_remaining_requests(self) -> int:
        """Get number of remaining API requests"""
        return max(0, self.max_requests - self.requests_made)

    def reset_request_counter(self):
        """Reset request counter (use carefully)"""
        self.requests_made = 0


class FlightDataCollector:
    """Orchestrates flight data collection using Amadeus API"""

    def __init__(self, amadeus_api: AmadeusAPI, database_manager):
        self.api = amadeus_api
        self.db = database_manager
        self.collection_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'total_flights_collected': 0,
            'routes_processed': 0
        }

    def collect_route_data(self, routes: List[Dict], dates: List[str],
                           max_dates_per_route: int = 3) -> Dict:
        """Collect flight data for multiple routes and dates"""

        print(f"Starting data collection for {len(routes)} routes and {len(dates)} dates")
        print("=" * 60)

        # Test API connection first
        if not self.api.test_api_connection():
            print("WARNING: Amadeus API not working. Switching to fallback mode.")
            print("This will generate simulated data for demonstration.")
            return self._generate_fallback_data(routes, dates)

        print("Amadeus API is working! Starting real data collection...")
        print("=" * 60)

        for route in routes:
            if self.api.get_remaining_requests() <= 0:
                print("API request limit reached")
                break

            origin = route['origin']
            destination = route['destination']
            priority = route.get('priority', 1)

            print(f"\nRoute: {origin} -> {destination} (Priority: {priority})")
            print("-" * 40)

            # Limit dates per route based on priority
            route_dates = dates[:max_dates_per_route] if priority == 1 else dates[:2]

            route_flights = []

            for date in route_dates:
                if self.api.get_remaining_requests() <= 0:
                    break

                # Make API request
                search_result = self.api.search_flights(
                    origin=origin,
                    destination=destination,
                    departure_date=date,
                    max_results=20
                )

                self.collection_stats['total_requests'] += 1

                if search_result and 'data' in search_result:
                    # Parse flight data
                    flights = self.api.parse_flight_offers(search_result, route, date)
                    route_flights.extend(flights)

                    self.collection_stats['successful_requests'] += 1
                    self.collection_stats['total_flights_collected'] += len(flights)

                    print(f"  Date {date}: {len(flights)} flights found")
                else:
                    print(f"  Date {date}: No data found")

                # Rate limiting
                time.sleep(1)

            # Save route data to database
            if route_flights:
                saved_count = self.db.save_flights(route_flights, 'amadeus')
                print(f"  Saved {saved_count} flights to database")
            else:
                print(f"  No flights collected for this route")

            self.collection_stats['routes_processed'] += 1
            print(f"  Remaining API requests: {self.api.get_remaining_requests()}")

        print("\n" + "=" * 60)
        print("DATA COLLECTION COMPLETED")
        print("=" * 60)
        print(f"Total requests made: {self.collection_stats['total_requests']}")
        print(f"Successful requests: {self.collection_stats['successful_requests']}")
        print(f"Total flights collected: {self.collection_stats['total_flights_collected']}")
        print(f"Routes processed: {self.collection_stats['routes_processed']}")

        return self.collection_stats

    def _generate_fallback_data(self, routes: List[Dict], dates: List[str]) -> Dict:
        """Generate simulated data when API is not working"""
        import numpy as np
        import random

        print("Generating simulated flight data...")
        np.random.seed(42)
        random.seed(42)

        airlines = [
            {'code': 'AA', 'name': 'American Airlines'},
            {'code': 'BA', 'name': 'British Airways'},
            {'code': 'LH', 'name': 'Lufthansa'},
            {'code': 'AF', 'name': 'Air France'},
            {'code': 'UA', 'name': 'United Airlines'},
            {'code': 'DL', 'name': 'Delta Air Lines'}
        ]

        all_flights = []

        for route in routes[:5]:
            origin = route['origin']
            destination = route['destination']

            print(f"Generating data for {origin}->{destination}")

            # Base prices by route type
            base_prices = {
                ('NYC', 'LON'): 650, ('NYC', 'PAR'): 680, ('LAX', 'NRT'): 850,
                ('NYC', 'LAX'): 320, ('LON', 'FRA'): 180, ('PAR', 'ROM'): 200
            }

            base_price = base_prices.get((origin, destination), 500)

            for date in dates[:2]:
                for i in range(8):
                    airline = random.choice(airlines)
                    hour = 6 + i * 2

                    flight_data = {
                        'origin': origin,
                        'destination': destination,
                        'departure_date': date,
                        'departure_time': f"{hour:02d}:{random.choice([0, 30]):02d}",
                        'arrival_time': f"{(hour + random.randint(2, 12)):02d}:{random.choice([0, 30]):02d}",
                        'airline_code': airline['code'],
                        'airline_name': airline['name'],
                        'flight_number': f"{airline['code']}{random.randint(1000, 9999)}",
                        'aircraft_type': random.choice(['737', 'A320', '777', 'A350']),
                        'duration_minutes': random.randint(120, 480),
                        'stops': random.choice([0, 1], p=[0.7, 0.3]),
                        'price_usd': round(base_price * (0.8 + random.random() * 0.4)),
                        'currency': 'USD',
                        'booking_class': random.choice(['Economy', 'Business', 'First']),
                        'seats_available': random.randint(1, 20),
                        'raw_data': {}
                    }

                    all_flights.append(flight_data)

            print(
                f"  Generated {len([f for f in all_flights if f['origin'] == origin and f['destination'] == destination])} flights")

        # Save to database
        if all_flights:
            saved_count = self.db.save_flights(all_flights, 'simulated')
            print(f"Saved {saved_count} simulated flights to database")

        return {
            'total_requests': len(routes) * 2,
            'successful_requests': len(routes) * 2,
            'total_flights_collected': len(all_flights),
            'routes_processed': len(routes)
        }

    def get_collection_summary(self) -> Dict:
        """Get summary of data collection process"""
        return {
            **self.collection_stats,
            'remaining_requests': self.api.get_remaining_requests(),
            'success_rate': (
                    self.collection_stats['successful_requests'] /
                    max(1, self.collection_stats['total_requests'])
            )
        }