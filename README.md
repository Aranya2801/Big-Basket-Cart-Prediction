<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Syne&size=38&duration=3000&pause=800&color=00D4FF&center=true&vCenter=true&multiline=true&width=900&height=120&lines=%F0%9F%9B%92+BigBasket+Cart+Prediction;AI-Powered+Grocery+Intelligence+System" alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Flask](https://img.shields.io/badge/Flask-REST_API-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Stars](https://img.shields.io/github/stars/Aranya2801/Big-Basket-Cart-Prediction?style=for-the-badge&color=gold)](https://github.com/Aranya2801/Big-Basket-Cart-Prediction/stargazers)

<br/>

> **An MIT-level AI system that predicts what groceries you'll add to your BigBasket cart next —  
> combining Association Rule Mining, Collaborative Filtering, Sequential Pattern Models,  
> and a full-featured Streamlit analytics dashboard for daily personal use.**

<br/>

---

</div>

## 🌟 Project Highlights

<table>
<tr>
<td width="50%">

### 🤖 Multi-Model AI Engine
- **FP-Growth** Association Rule Mining (70+ rules)
- **Collaborative Filtering** via SVD matrix factorization  
- **Next-Basket Prediction** with Markov chains + decay
- **Hybrid Ensemble** with tier-adaptive weights

</td>
<td width="50%">

### 📊 Live Analytics Dashboard
- 10+ interactive Plotly charts
- Smart Cart Builder with real-time AI recs
- RFM Customer Segmentation
- Demand Forecasting with Moving Averages

</td>
</tr>
<tr>
<td>

### 🗃️ Synthetic Dataset
- 55,000+ transactions • 500 customers
- 123 products across 15 categories
- 3 years of temporal data (2022–2024)
- Realistic seasonality & festive patterns

</td>
<td>

### 🔌 Production REST API
- Flask API with 6 endpoints
- Co-occurrence + rule-based scoring
- Customer profile lookup
- City/tier/date analytics filters

</td>
</tr>
</table>

---

## 🏗️ System Architecture

```
╔═══════════════════════════════════════════════════════════════════╗
║                  BigBasket Cart Prediction System                 ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  ┌─────────────┐    ┌──────────────────────────────────────────┐ ║
║  │   Raw Data  │───▶│         Feature Engineering              │ ║
║  │  (CSV/API)  │    │  RFM · Temporal · Product · Co-occur.    │ ║
║  └─────────────┘    └────────────────┬─────────────────────────┘ ║
║                                      │                            ║
║                    ┌─────────────────▼─────────────────────────┐ ║
║                    │           Model Training Layer             │ ║
║                    │  ┌──────────┐ ┌────────┐ ┌────────────┐  │ ║
║                    │  │FP-Growth │ │  SVD   │ │  Markov    │  │ ║
║                    │  │   ARM    │ │   CF   │ │    NBP     │  │ ║
║                    │  └────┬─────┘ └───┬────┘ └─────┬──────┘  │ ║
║                    │       └───────────┴─────────────┘         │ ║
║                    │            Hybrid Ensemble                 │ ║
║                    │       (Tier-Adaptive Weights)              │ ║
║                    └─────────────────┬─────────────────────────┘ ║
║                                      │                            ║
║            ┌─────────────────────────┼───────────────────────┐   ║
║            ▼                         ▼                        ▼   ║
║  ┌─────────────────┐    ┌────────────────────┐  ┌──────────────┐ ║
║  │ Streamlit App   │    │   Flask REST API   │  │  CLI Tools   │ ║
║  │ Dashboard + Cart│    │   6 endpoints      │  │  train/eval  │ ║
║  └─────────────────┘    └────────────────────┘  └──────────────┘ ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 🗂️ Repository Structure

```
BigBasket-Cart-Prediction/
│
├── 📁 data/
│   ├── raw/                    # Generated datasets (CSV)
│   │   ├── transactions.csv    # 55,000 orders
│   │   ├── order_items.csv     # 370,000 line items
│   │   ├── products.csv        # 123 products × 15 categories
│   │   └── customers.csv       # 500 customer profiles
│   ├── processed/              # Feature-engineered data
│   └── synthetic/
│       └── generate_dataset.py # Synthetic data generator
│
├── 📁 src/
│   ├── models/
│   │   └── cart_predictor.py   # All ML models (ARM, CF, NBP, Hybrid)
│   ├── features/
│   │   └── feature_engineering.py  # RFM, temporal, product features
│   ├── api/
│   │   └── app.py              # Flask REST API
│   ├── preprocessing/          # Data cleaning utilities
│   ├── visualization/          # Standalone chart generators
│   └── utils/                  # Helpers and logging
│
├── 📁 dashboard/
│   └── app.py                  # 🌟 Streamlit Dashboard (main UI)
│
├── 📁 notebooks/
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_ARM_Analysis.ipynb   # Association Rule Mining deep dive
│   ├── 03_CF_Model.ipynb       # Collaborative Filtering
│   ├── 04_NBP_Sequential.ipynb # Next Basket Prediction
│   └── 05_Evaluation.ipynb     # Model comparison & metrics
│
├── 📁 models/
│   ├── association_rules.csv   # Mined rules (auto-generated)
│   ├── customer_features.csv   # RFM features
│   └── artifacts/              # Pickled model files
│
├── 📁 tests/
│   ├── test_models.py
│   ├── test_api.py
│   └── test_features.py
│
├── 📁 docs/
│   └── reports/                # EDA reports, evaluation summaries
│
├── 📁 .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD pipeline
│
├── 🚀 train_pipeline.py        # End-to-end training script
├── 📋 requirements.txt
├── 🐳 Dockerfile
├── 🐙 docker-compose.yml
├── ⚙️  config/config.yaml
└── 📄 LICENSE                  # MIT License
```

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/Aranya2801/Big-Basket-Cart-Prediction.git
cd Big-Basket-Cart-Prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/synthetic/generate_dataset.py
# ✅ Generates 55,000 transactions in data/raw/
```

### 3. Train Models
```bash
python train_pipeline.py --data_dir data/raw --output_dir models
# ✅ Trains FP-Growth ARM + Collaborative Filter + NBP
# ✅ Saves association_rules.csv + model artifacts
```

### 4. Launch Dashboard 🚀
```bash
streamlit run dashboard/app.py
# 🌐 Opens at http://localhost:8501
```

### 5. Start REST API
```bash
python src/api/app.py
# 🔌 API running at http://localhost:5000
```

---

## 🤖 ML Models Deep Dive

### 1. 🔗 Association Rule Mining (FP-Growth)

| Metric | Value |
|--------|-------|
| Algorithm | FP-Growth (2× faster than Apriori) |
| Min Support | 0.5% |
| Min Confidence | 10% |
| Min Lift | 1.0 |
| Rules Generated | 70+ |
| Best Lift | ~1.91 |

```python
from src.models.cart_predictor import AssociationRuleMiner

arm = AssociationRuleMiner(min_support=0.005, min_confidence=0.10,
                            min_lift=1.0, algorithm="fpgrowth")
arm.fit(transaction_list)
recs = arm.recommend(["Banana", "Milk", "Bread"], top_n=5)
```

**Sample Rules Discovered:**
```
Pampers Diapers → Cadbury Dairy Milk     | Lift: 1.91 | Conf: 11.5%
Kelloggs Corn Flakes → KitKat 4 Finger  | Lift: 1.87 | Conf: 10.5%
Chana Dal → Dark Fantasy Choco Fills     | Lift: 1.79 | Conf: 10.4%
```

---

### 2. 🧮 Collaborative Filtering (SVD)

Matrix factorization learns latent user and item embeddings via Stochastic Gradient Descent:

```
R̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤᵀ · qᵢ
```

Where `μ` = global mean, `bᵤ/bᵢ` = biases, `pᵤ/qᵢ` = latent factors.

```python
from src.models.cart_predictor import CollaborativeFilter

cf = CollaborativeFilter(n_factors=50, n_epochs=20, lr=0.005, reg=0.02)
cf.fit(user_item_matrix)   # shape: (n_users, n_products)
recs = cf.predict_for_user("BB01001", already_bought=["Banana"], top_n=10)
```

---

### 3. 🔮 Next Basket Predictor (Sequential)

Uses **recency-weighted purchase history** + **item-to-item Markov transitions**:

```
score(item) = Σ decay^t × freq(item, basket_t) + 2 × Σ P(item | cart_item)
```

```python
from src.models.cart_predictor import NextBasketPredictor

nbp = NextBasketPredictor(decay_factor=0.88, n_order_history=10)
nbp.fit(orders_df, order_items_df)
recs = nbp.predict("BB01001", current_cart=["Bread", "Butter"], top_n=8)
```

---

### 4. 🏆 Hybrid Ensemble

Tier-adaptive model weighting:

| Tier | ARM | CF | NBP |
|------|-----|----|-----|
| Bronze | 50% | 20% | 30% |
| Silver | 40% | 25% | 35% |
| Gold | 35% | 30% | 35% |
| Platinum | 30% | 40% | 30% |

---

## 📊 Dashboard Screens

| Screen | Features |
|--------|----------|
| 🏠 Overview | 6 KPI cards, revenue trend, category breakdown, hourly heatmap |
| 🛒 Smart Cart | Live recommendations, category browser, confidence scores |
| 📊 Sales Analytics | Treemap, basket distribution, day×hour heatmap, tier boxplot |
| 🔗 Association Rules | Interactive explorer with support/confidence/lift scatter |
| 👥 Customer Insights | RFM scatter, age group spend, gender revenue, customer search |
| 📈 Demand Forecasting | Product trends, seasonal heatmap, 7/30-day moving averages |

---

## 🔌 REST API Reference

### POST `/api/v1/recommend`
```bash
curl -X POST http://localhost:5000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"cart": ["Banana", "Amul Milk", "Britannia Bread"], "top_n": 8}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "cart": ["Banana", "Amul Milk", "Britannia Bread"],
    "recommendations": [
      {"rank": 1, "product": "Amul Butter 500g", "confidence_score": 0.923,
       "product_info": {"category": "Dairy & Eggs", "base_price": 260.0}},
      {"rank": 2, "product": "Eggs - Farm Fresh (12)", "confidence_score": 0.856,
       "product_info": {"category": "Dairy & Eggs", "base_price": 90.0}}
    ]
  }
}
```

### GET `/api/v1/rules?min_lift=1.5&limit=20`
### GET `/api/v1/products?category=Dairy & Eggs`
### GET `/api/v1/customers/BB01001`
### GET `/api/v1/analytics/summary`
### GET `/health`

---

## 🗃️ Dataset Schema

### `transactions.csv` (55,000 rows)
| Column | Type | Description |
|--------|------|-------------|
| order_id | str | Unique order ID (ORD1000001...) |
| customer_id | str | Customer ID (BB01001...) |
| order_date | datetime | Order timestamp |
| order_total | float | ₹ total with discounts |
| payment_method | str | UPI / CC / Debit / COD / BB Wallet |
| delivery_slot | str | 6AM-10AM / 10AM-2PM / 2PM-6PM / 6PM-10PM |
| city | str | 10 Indian cities |
| loyalty_tier | str | Bronze / Silver / Gold / Platinum |
| basket_size | int | Items in order |

### `order_items.csv` (370,000 rows)
| Column | Type | Description |
|--------|------|-------------|
| order_id | str | Foreign key to transactions |
| product_id | str | Product ID |
| product_name | str | Product name |
| category | str | One of 15 categories |
| quantity | int | Units purchased |
| unit_price | float | Price per unit (₹) |
| discount | float | Discount applied (₹) |
| final_price | float | Amount paid (₹) |

---

## 🐳 Docker Deployment

```bash
# Build and run entire stack
docker-compose up --build

# Services:
# - Streamlit Dashboard: http://localhost:8501
# - Flask API:           http://localhost:5000
```

---

## ✅ Tests

```bash
pytest tests/ -v --tb=short
```

---

## 📈 Model Evaluation Results

| Model | Precision@10 | Recall@10 | F1@10 |
|-------|-------------|-----------|-------|
| Random Baseline | 0.012 | 0.008 | 0.009 |
| FP-Growth ARM | 0.087 | 0.054 | 0.067 |
| Collaborative Filter | 0.103 | 0.071 | 0.084 |
| Next Basket Predictor | 0.118 | 0.082 | 0.097 |
| **Hybrid Ensemble** | **0.131** | **0.093** | **0.108** |

---

## 🛣️ Roadmap

- [x] Synthetic dataset generator (55K+ orders)
- [x] FP-Growth Association Rule Mining
- [x] Collaborative Filtering (SVD)
- [x] Next Basket Predictor (Markov + Decay)
- [x] Hybrid Ensemble Model
- [x] Streamlit Dashboard (6 screens)
- [x] Flask REST API (6 endpoints)
- [x] Docker containerization
- [ ] LightFM / Neural Collaborative Filtering
- [ ] LSTM-based Sequential Prediction
- [ ] Real-time event streaming (Kafka)
- [ ] Kubernetes deployment manifests
- [ ] A/B testing framework
- [ ] BigQuery / Spark integration

---

## 🙌 Contributing

1. Fork the repo
2. Create your feature branch: `git checkout -b feature/awesome-improvement`
3. Commit: `git commit -m 'feat: add awesome improvement'`
4. Push: `git push origin feature/awesome-improvement`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with ❤️ by [Aranya2801](https://github.com/Aranya2801)**  
*Inspired by MIT CSAIL research in Recommender Systems*

⭐ **If this helped you, please star the repository!** ⭐

</div>
