import sqlite3
import json
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime
import os


class FlightDatabase:
    """Database manager for flight search optimization data"""

    def __init__(self, db_path: str = "data/flights.db"):
        self.db_path = db_path
        self._create_database_directory()
        self.init_database()

    def _create_database_directory(self):
        """Create database directory if it doesn't exist"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

    def init_database(self):
        """Initialize database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Flights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS flights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin TEXT NOT NULL,
                destination TEXT NOT NULL,
                departure_date TEXT NOT NULL,
                departure_time TEXT,
                arrival_time TEXT,
                airline_code TEXT,
                airline_name TEXT,
                flight_number TEXT,
                aircraft_type TEXT,
                duration_minutes INTEGER,
                stops INTEGER DEFAULT 0,
                price_usd REAL,
                currency TEXT DEFAULT 'USD',
                booking_class TEXT,
                seats_available INTEGER,
                source TEXT NOT NULL,
                raw_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Search sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS search_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                origin TEXT NOT NULL,
                destination TEXT NOT NULL,
                search_date TEXT NOT NULL,
                results_count INTEGER,
                search_duration_ms INTEGER,
                user_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Airlines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS airlines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                airline_code TEXT UNIQUE NOT NULL,
                airline_name TEXT,
                icao_code TEXT,
                country TEXT,
                rating REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # A/B Test results table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ab_test_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_name TEXT NOT NULL,
                variant TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                sample_size INTEGER,
                confidence_level REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_flights_route ON flights(origin, destination)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_flights_date ON flights(departure_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_flights_price ON flights(price_usd)')

        conn.commit()
        conn.close()

    def save_flights(self, flights_data: List[Dict], source: str) -> int:
        """Save flight data to database"""
        if not flights_data:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        saved_count = 0
        for flight in flights_data:
            try:
                cursor.execute('''
                    INSERT INTO flights (
                        origin, destination, departure_date, departure_time,
                        arrival_time, airline_code, airline_name, flight_number,
                        aircraft_type, duration_minutes, stops, price_usd,
                        currency, booking_class, seats_available, source, raw_data
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    flight.get('origin'),
                    flight.get('destination'),
                    flight.get('departure_date'),
                    flight.get('departure_time'),
                    flight.get('arrival_time'),
                    flight.get('airline_code'),
                    flight.get('airline_name'),
                    flight.get('flight_number'),
                    flight.get('aircraft_type'),
                    flight.get('duration_minutes'),
                    flight.get('stops', 0),
                    flight.get('price_usd'),
                    flight.get('currency', 'USD'),
                    flight.get('booking_class'),
                    flight.get('seats_available'),
                    source,
                    json.dumps(flight.get('raw_data', {}))
                ))
                saved_count += 1
            except sqlite3.Error as e:
                print(f"Error saving flight data: {e}")
                continue

        conn.commit()
        conn.close()
        return saved_count

    def save_airlines(self, airlines_data: List[Dict]) -> int:
        """Save airline data to database"""
        if not airlines_data:
            return 0

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        saved_count = 0
        for airline in airlines_data:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO airlines (
                        airline_code, airline_name, icao_code, country, rating
                    ) VALUES (?, ?, ?, ?, ?)
                ''', (
                    airline.get('airline_code'),
                    airline.get('airline_name'),
                    airline.get('icao_code'),
                    airline.get('country'),
                    airline.get('rating')
                ))
                saved_count += 1
            except sqlite3.Error as e:
                print(f"Error saving airline data: {e}")
                continue

        conn.commit()
        conn.close()
        return saved_count

    def get_flights_by_route(self, origin: str, destination: str,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None) -> pd.DataFrame:
        """Get flights data for specific route"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT * FROM flights 
            WHERE origin = ? AND destination = ?
        '''
        params = [origin, destination]

        if start_date:
            query += ' AND departure_date >= ?'
            params.append(start_date)

        if end_date:
            query += ' AND departure_date <= ?'
            params.append(end_date)

        query += ' ORDER BY departure_date, price_usd'

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df

    def get_route_statistics(self) -> pd.DataFrame:
        """Get statistics for all routes"""
        conn = sqlite3.connect(self.db_path)

        query = '''
            SELECT 
                origin,
                destination,
                COUNT(*) as flight_count,
                AVG(price_usd) as avg_price,
                MIN(price_usd) as min_price,
                MAX(price_usd) as max_price,
                AVG(duration_minutes) as avg_duration,
                COUNT(DISTINCT airline_code) as airline_count
            FROM flights
            WHERE price_usd IS NOT NULL
            GROUP BY origin, destination
            ORDER BY flight_count DESC
        '''

        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

    def save_ab_test_result(self, test_name: str, variant: str,
                            metric_name: str, metric_value: float,
                            sample_size: int, confidence_level: float = 0.95):
        """Save A/B test result to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO ab_test_results (
                test_name, variant, metric_name, metric_value,
                sample_size, confidence_level
            ) VALUES (?, ?, ?, ?, ?, ?)
        ''', (test_name, variant, metric_name, metric_value, sample_size, confidence_level))

        conn.commit()
        conn.close()

    def get_database_stats(self) -> Dict[str, int]:
        """Get overall database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        stats = {}

        # Total flights
        cursor.execute('SELECT COUNT(*) FROM flights')
        stats['total_flights'] = cursor.fetchone()[0]

        # Unique routes
        cursor.execute('SELECT COUNT(DISTINCT origin || "-" || destination) FROM flights')
        stats['unique_routes'] = cursor.fetchone()[0]

        # Unique airlines
        cursor.execute('SELECT COUNT(DISTINCT airline_code) FROM flights WHERE airline_code IS NOT NULL')
        stats['unique_airlines'] = cursor.fetchone()[0]

        # Data sources
        cursor.execute('SELECT source, COUNT(*) FROM flights GROUP BY source')
        sources = cursor.fetchall()
        for source, count in sources:
            stats[f'flights_from_{source}'] = count

        conn.close()
        return stats
