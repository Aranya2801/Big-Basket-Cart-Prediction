# 📋 Changelog

All notable changes to this project are documented here.

## [2.0.0] — 2024-12-01

### Added
- **Hybrid Ensemble Model** combining ARM + CF + NBP with tier-adaptive weights
- **Next Basket Predictor** using Markov chain + recency decay scoring
- **Collaborative Filtering** via SVD matrix factorization with SGD training
- **Streamlit Dashboard** with 6 fully interactive screens
- **Flask REST API** with 6 production-ready endpoints
- **CLI Tool** (`bb_cli.py`) for daily interactive use
- **Synthetic Dataset Generator** producing 55K+ realistic transactions
- **Feature Engineering** module: RFM, temporal, product, reorder label features
- **Docker + Docker Compose** containerization
- **GitHub Actions CI/CD** pipeline
- **Unit Test Suite** (25 tests across models, features, integration)
- **5 Jupyter Notebooks** (EDA, ARM, CF, NBP, Evaluation)

### Changed
- Upgraded minimum Python to 3.11
- Replaced basic Apriori with faster FP-Growth as default algorithm
- Streamlit theme upgraded to full dark mode with CSS variables

### Fixed
- Rule generation now correctly handles single-item frequent sets
- Co-occurrence index properly excludes cart items from recommendations

---

## [1.0.0] — 2024-06-01

### Added
- Initial market basket analysis with Apriori
- Basic product recommendation engine
- Simple EDA notebook

---

## Roadmap

### [2.1.0] — Planned
- LightFM integration for hybrid collaborative + content filtering
- LSTM sequential recommendation model
- Real-time Kafka event streaming
- A/B testing framework for recommendation strategies

### [3.0.0] — Planned
- Kubernetes deployment manifests
- BigQuery / Spark integration for large-scale data
- BERT4Rec / SASRec transformer-based recommendations
- Multi-armed bandit exploration for cold-start users
