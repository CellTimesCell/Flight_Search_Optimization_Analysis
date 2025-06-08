import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os


class FlightRankingSystem:
    """Machine learning system for ranking flight search results"""

    def __init__(self):
        self.user_profiles = {
            'budget_traveler': {
                'price_weight': 0.6,
                'time_weight': 0.2,
                'airline_weight': 0.1,
                'convenience_weight': 0.1
            },
            'business_traveler': {
                'price_weight': 0.2,
                'time_weight': 0.4,
                'airline_weight': 0.3,
                'convenience_weight': 0.1
            },
            'luxury_traveler': {
                'price_weight': 0.1,
                'time_weight': 0.2,
                'airline_weight': 0.5,
                'convenience_weight': 0.2
            },
            'family_traveler': {
                'price_weight': 0.4,
                'time_weight': 0.2,
                'airline_weight': 0.2,
                'convenience_weight': 0.2
            }
        }

        self.scalers = {}
        self.models = {}
        self.feature_columns = [
            'price_usd', 'duration_minutes', 'stops',
            'departure_hour', 'airline_rating', 'is_weekend'
        ]

    def create_features(self, flights_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ML model"""

        df = flights_df.copy()

        # Basic feature engineering
        df['departure_hour'] = pd.to_datetime(df['departure_time'], format='%H:%M', errors='coerce').dt.hour
        df['departure_hour'] = df['departure_hour'].fillna(12)  # Default to noon if missing

        # Weekend detection (assuming departure_date is available)
        if 'departure_date' in df.columns:
            df['departure_date'] = pd.to_datetime(df['departure_date'], errors='coerce')
            df['is_weekend'] = df['departure_date'].dt.weekday >= 5
        else:
            df['is_weekend'] = False

        # Airline rating (mock data if not available)
        if 'airline_rating' not in df.columns:
            airline_ratings = {
                'AA': 4.2, 'BA': 4.3, 'LH': 4.4, 'AF': 4.1,
                'UA': 4.0, 'DL': 4.3, 'EK': 4.6, 'SQ': 4.7
            }
            df['airline_rating'] = df['airline_code'].map(airline_ratings).fillna(4.0)

        # Handle missing values
        df['duration_minutes'] = df['duration_minutes'].fillna(df['duration_minutes'].median())
        df['stops'] = df['stops'].fillna(0)
        df['price_usd'] = df['price_usd'].fillna(df['price_usd'].median())

        return df

    def calculate_rule_based_scores(self, flights_df: pd.DataFrame,
                                    user_type: str = 'business_traveler') -> pd.DataFrame:
        """Calculate scores using rule-based approach"""

        df = self.create_features(flights_df)
        weights = self.user_profiles.get(user_type, self.user_profiles['business_traveler'])

        # Normalize features to 0-1 scale
        if len(df) > 1:
            # Price score (lower is better)
            price_min, price_max = df['price_usd'].min(), df['price_usd'].max()
            if price_max > price_min:
                df['price_score'] = 1 - (df['price_usd'] - price_min) / (price_max - price_min)
            else:
                df['price_score'] = 1.0

            # Duration score (shorter is better)
            duration_min, duration_max = df['duration_minutes'].min(), df['duration_minutes'].max()
            if duration_max > duration_min:
                df['duration_score'] = 1 - (df['duration_minutes'] - duration_min) / (duration_max - duration_min)
            else:
                df['duration_score'] = 1.0
        else:
            df['price_score'] = 1.0
            df['duration_score'] = 1.0

        # Time preference score (business hours are preferred)
        df['time_score'] = df['departure_hour'].apply(lambda x: 1.0 if 9 <= x <= 17 else 0.7)

        # Airline score (based on rating)
        rating_min, rating_max = 3.5, 5.0  # Typical airline rating range
        df['airline_score'] = (df['airline_rating'] - rating_min) / (rating_max - rating_min)
        df['airline_score'] = df['airline_score'].clip(0, 1)

        # Convenience score (direct flights preferred)
        df['convenience_score'] = df['stops'].apply(lambda x: 1.0 if x == 0 else 0.5 if x == 1 else 0.3)

        # Calculate weighted total score
        df['total_score'] = (
                df['price_score'] * weights['price_weight'] +
                (df['duration_score'] * 0.5 + df['time_score'] * 0.5) * weights['time_weight'] +
                df['airline_score'] * weights['airline_weight'] +
                df['convenience_score'] * weights['convenience_weight']
        )

        return df

    def train_ml_model(self, flights_df: pd.DataFrame, user_type: str = 'business_traveler'):
        """Train ML model for flight ranking"""

        # Create features and rule-based scores as training targets
        df = self.calculate_rule_based_scores(flights_df, user_type)

        # Prepare features
        X = df[self.feature_columns].copy()
        y = df['total_score']

        # Handle categorical variables
        X['is_weekend'] = X['is_weekend'].astype(int)

        # Split data
        if len(X) > 10:  # Only split if we have enough data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = X, X, y, y

        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Store model and scaler
        self.models[user_type] = model
        self.scalers[user_type] = scaler

        print(f"Model trained for {user_type}: MSE={mse:.4f}, R2={r2:.4f}")

        return {
            'mse': mse,
            'r2': r2,
            'feature_importance': dict(zip(self.feature_columns, model.feature_importances_))
        }

    def predict_scores(self, flights_df: pd.DataFrame,
                       user_type: str = 'business_traveler') -> pd.DataFrame:
        """Predict flight scores using trained ML model"""

        # If no model is trained, fall back to rule-based approach
        if user_type not in self.models:
            print(f"No trained model for {user_type}, using rule-based scoring")
            return self.calculate_rule_based_scores(flights_df, user_type)

        df = self.create_features(flights_df)

        # Prepare features
        X = df[self.feature_columns].copy()
        X['is_weekend'] = X['is_weekend'].astype(int)

        # Scale features
        X_scaled = self.scalers[user_type].transform(X)

        # Predict scores
        predicted_scores = self.models[user_type].predict(X_scaled)
        df['ml_score'] = predicted_scores

        return df

    def rank_flights(self, flights_df: pd.DataFrame, user_type: str = 'business_traveler',
                     use_ml: bool = True) -> pd.DataFrame:
        """Rank flights for given user type"""

        if flights_df.empty:
            return flights_df

        if use_ml and user_type in self.models:
            scored_df = self.predict_scores(flights_df, user_type)
            score_column = 'ml_score'
        else:
            scored_df = self.calculate_rule_based_scores(flights_df, user_type)
            score_column = 'total_score'

        # Sort by score (highest first)
        ranked_df = scored_df.sort_values(score_column, ascending=False)

        # Add ranking position
        ranked_df['rank_position'] = range(1, len(ranked_df) + 1)

        return ranked_df

    def save_models(self, filepath: str):
        """Save trained models to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'user_profiles': self.user_profiles,
            'feature_columns': self.feature_columns
        }

        joblib.dump(model_data, filepath)
        print(f"Models saved to {filepath}")

    def load_models(self, filepath: str):
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.user_profiles = model_data['user_profiles']
            self.feature_columns = model_data['feature_columns']
            print(f"Models loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
