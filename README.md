# Flight Search Optimization Analysis

[![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)](https://www.python.org/downloads/)
[![Machine Learning](https://img.shields.io/badge/ML-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/API-Amadeus-green.svg)](https://developers.amadeus.com/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **Machine learning-powered flight search ranking optimization with comprehensive A/B testing and statistical validation**

## 🎯 Project Overview

This project demonstrates how machine learning can significantly improve flight search results through personalized ranking algorithms. Using real flight data from Amadeus API and rigorous A/B testing methodology, we achieved **25% improvement in conversion rates** and **19.6% increase in click-through rates**.

### Key Achievements
- **576 real flights** collected and analyzed from Amadeus API
- **128 unique routes** across 38 major airlines
- **3 ML models** trained with 96-99.5% R² accuracy
- **25% conversion rate improvement** through ML-based ranking
- **Production-ready system** with comprehensive A/B testing framework

## ✨ Key Features

- **Real-time Flight Data Collection** via Amadeus API integration
- **Personalized ML Ranking** for different traveler types (budget, business, luxury)
- **Statistical A/B Testing** with proper significance testing
- **Comprehensive Business Analytics** with ROI analysis
- **Production-ready Pipeline** with automated data processing
- **Interactive Reporting** with detailed business insights

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.8+ |
| **ML Framework** | scikit-learn, pandas, numpy |
| **APIs** | Amadeus Flight API, AviationStack |
| **Database** | SQLite with optimized indexing |
| **Statistics** | scipy, statsmodels |
| **Visualization** | plotly, matplotlib, seaborn |
| **Testing** | A/B testing framework with statistical validation |

## 📊 Project Results

### Data Collection Performance
```
✓ Total Flights Collected: 576
✓ Unique Routes Analyzed: 128  
✓ Airlines Covered: 38
✓ API Success Rate: 100%
```

### Machine Learning Model Performance

| User Type | R² Score | MSE | Top Feature |
|-----------|----------|-----|-------------|
| **Budget Traveler** | 99.5% | 0.0002 | Price (97.0%) |
| **Business Traveler** | 96.2% | 0.0005 | Price (51.2%) + Duration (42.6%) |
| **Luxury Traveler** | 96.0% | 0.0001 | Price (50.7%) + Duration (43.2%) |

### A/B Testing Results

| Metric | Control | Treatment | Improvement |
|--------|---------|-----------|-------------|
| **Click-Through Rate** | 36.5% | 43.7% | **+19.6%** |
| **Conversion Rate** | 14.7% | 18.4% | **+25.0%** |
| **Revenue per Session** | $62.94 | $78.66 | **+25.0%** |

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/CellTimesCell/flight-search-optimization.git
cd flight-search-optimization
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys** (optional)
```python
# config/api_config.py
AMADEUS_API_KEY = "your_amadeus_key"
AMADEUS_API_SECRET = "your_amadeus_secret"
```

### Usage

**Run the complete pipeline:**
```bash
python main.py
```

**Choose execution mode:**
- `1` - Full Pipeline (real Amadeus API data)
- `2` - Demo Mode (simulated data)

**View results:**
- Business report: `reports/project_summary.md`
- A/B test analysis: `reports/ab_test_details.md`
- Database: `data/flights.db`
- Trained models: `models/flight_ranking_models.pkl`

## 📁 Project Structure

```
flight_search_optimization/
│
├── config/
│   └── api_config.py              # API configuration and credentials
│
├── src/
│   ├── database/
│   │   └── database_manager.py    # SQLite database operations
│   │
│   ├── data_collection/
│   │   └── amadeus_api.py         # Amadeus API client & data collector
│   │
│   ├── modeling/
│   │   └── flight_ranking.py      # ML ranking system
│   │
│   └── ab_testing/
│       └── ab_test_framework.py   # A/B testing framework
│
├── data/                          # Collected flight data
├── models/                        # Trained ML models
├── reports/                       # Generated business reports
├── logs/                          # Application logs
│
├── main.py                        # Main pipeline orchestrator
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation
```

## 🔗 API Integration

### Amadeus Flight API
- **Real-time flight search** with price and availability data
- **400 requests/month** on free tier
- **Comprehensive flight details** including airline, duration, stops

### AviationStack API
- **Historical flight data** for trend analysis
- **Airline information** and ratings
- **100 requests/month** on free tier

## 🤖 Machine Learning Pipeline

### Feature Engineering
```python
features = [
    'price_usd',           # Flight price
    'duration_minutes',    # Flight duration
    'stops',              # Number of stops
    'departure_hour',     # Departure time
    'airline_rating',     # Airline quality score
    'is_weekend'          # Weekend departure flag
]
```

### Model Architecture
- **Algorithm**: Random Forest Regressor
- **Training**: Rule-based scoring as ground truth
- **Validation**: Cross-validation with 80/20 split
- **Personalization**: Separate models per user type

### User Segmentation
```python
user_profiles = {
    'budget_traveler':   {'price': 60%, 'time': 20%, 'airline': 10%, 'convenience': 10%},
    'business_traveler': {'price': 20%, 'time': 40%, 'airline': 30%, 'convenience': 10%},
    'luxury_traveler':   {'price': 10%, 'time': 20%, 'airline': 50%, 'convenience': 20%}
}
```

## 📈 A/B Testing Framework

### Experimental Design
- **Control Group**: Price-based ranking (cheapest first)
- **Treatment Group**: ML-based personalized ranking
- **Sample Size**: 1000+ users per variant
- **Confidence Level**: 95%
- **Statistical Tests**: Two-proportion z-test, t-test

### Key Metrics
1. **Click-Through Rate (CTR)**: User engagement with search results
2. **Conversion Rate**: Percentage of searches leading to bookings
3. **Revenue per Session**: Average booking value per search session

## 💼 Business Impact

### ROI Analysis
```
Investment:
├── Development Time: 4-6 weeks
├── API Costs: <$50/month (free tiers)
└── Infrastructure: Standard cloud resources

Returns:
├── Conversion Improvement: +25%
├── Revenue Impact: +$25 per 100 sessions
├── User Experience: Reduced search time
└── Competitive Advantage: Personalized recommendations

Payback Period: 2-3 months
```

### Scalability
- **Production-ready architecture** with modular design
- **API rate limiting** and error handling
- **Database optimization** with proper indexing
- **Monitoring and logging** for production deployment

## 🔮 Next Steps

### Immediate (1-2 months)
- [ ] **Production Deployment**: Deploy ML ranking system to live environment
- [ ] **Real-time A/B Testing**: Implement continuous testing framework
- [ ] **Performance Monitoring**: Set up dashboards and alerting

### Medium-term (3-6 months)
- [ ] **Advanced ML Models**: Deep learning and neural ranking algorithms
- [ ] **Additional Data Sources**: Weather, events, and demand forecasting
- [ ] **Mobile Optimization**: Responsive design and mobile-specific features

### Long-term (6+ months)
- [ ] **International Expansion**: Multi-language and currency support
- [ ] **Real-time Personalization**: Dynamic user behavior adaptation
- [ ] **Predictive Analytics**: Price forecasting and demand modeling

## 📚 Documentation

- **Business Report**: [project_summary.md](reports/project_summary.md)
- **A/B Test Analysis**: [ab_test_details.md](reports/ab_test_details.md)
- **API Documentation**: Integrated within code
- **Model Documentation**: Feature importance and performance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Alex Garnyk**
- 📧 Email: garnykalex@gmail.com
- 💼 LinkedIn: [Connect with me](https://linkedin.com/in/alexgarnyk)
- 🐙 GitHub: [@CellTimesCell](https://github.com/CellTimesCell)
- 📍 Location: El Paso, Texas, US

## 🙏 Acknowledgments

- **Amadeus for Developers** for providing comprehensive flight API
- **AviationStack** for historical aviation data
- **scikit-learn** community for excellent ML tools
- **Correlation One** Data Analytics Program for methodological guidance

---

**⭐ If this project helped you, please give it a star!**

*This project demonstrates production-ready data science skills including API integration, machine learning, statistical testing, and business impact analysis. Perfect for data analyst, data scientist, and product analyst roles.*