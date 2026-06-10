# 🤝 Contributing to BigBasket Cart Prediction

Thank you for your interest in contributing! This guide will help you get started.

## 🚀 Getting Started

### Fork & Clone
```bash
git clone https://github.com/YOUR_USERNAME/Big-Basket-Cart-Prediction.git
cd Big-Basket-Cart-Prediction
git checkout -b feature/your-feature-name
```

### Setup Dev Environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"        # Install in editable mode
```

### Generate Data & Run Tests
```bash
python data/synthetic/generate_dataset.py
pytest tests/ -v
```

## 🧩 Project Areas

| Area | Files | Description |
|------|-------|-------------|
| ML Models | `src/models/cart_predictor.py` | ARM, CF, NBP, Hybrid |
| Features | `src/features/feature_engineering.py` | RFM, temporal features |
| Dashboard | `dashboard/app.py` | Streamlit UI |
| API | `src/api/app.py` | Flask REST endpoints |
| Tests | `tests/` | Pytest test suite |
| Data Gen | `data/synthetic/` | Synthetic dataset |

## 📋 Contribution Guidelines

### Code Style
- Follow **PEP 8** with max line length of 100 chars
- Use **type hints** on all function signatures
- Write **docstrings** for all public methods
- Run `black src/ dashboard/` before committing

### Commit Messages (Conventional Commits)
```
feat: add LSTM-based sequential prediction model
fix: resolve cold-start issue in collaborative filter
docs: update API reference with new endpoint examples
test: add unit tests for hybrid ensemble weights
perf: optimize FP-tree construction with sparse arrays
refactor: extract co-occurrence builder to utils module
```

### Pull Request Checklist
- [ ] Tests pass: `pytest tests/ -v`
- [ ] No lint errors: `flake8 src/ --max-line-length=100`
- [ ] Docstrings added/updated
- [ ] README updated if needed
- [ ] CHANGELOG entry added

## 🐛 Reporting Bugs

Please open an issue with:
1. **Environment**: Python version, OS, package versions
2. **Steps to reproduce**: Minimal code snippet
3. **Expected vs actual behavior**
4. **Error traceback** (full)

## 💡 Feature Requests

Open an issue tagged `enhancement` with:
- Use case description
- Proposed API / interface
- Any relevant papers or implementations

## 🔬 Research Contributions

This project welcomes academic contributions:
- New recommendation algorithms (BERT4Rec, SASRec, etc.)
- Better evaluation metrics
- Real dataset integration (with proper licensing)
- Benchmarking against state-of-the-art methods

## 📄 License

By contributing, you agree your contributions will be licensed under the MIT License.
