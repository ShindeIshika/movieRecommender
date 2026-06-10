# 🎬 Movie Recommendation Engine

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-green.svg)](https://fastapi.tiangolo.com)

**3 recommendation strategies + REST API + 75% Precision@10**

---

## 📊 Key Results

| Metric | Score |
|--------|-------|
| Precision@10 | **75.38%** |
| Recall@10 | **63.47%** |
| RMSE | **0.9255** |

---

## 🎯 What's Inside

| Approach | Technique | Use Case |
|----------|-----------|----------|
| Content-Based | TF-IDF + Cosine Similarity | Cold-start |
| Collaborative | User-User Similarity | Established users |
| Matrix Factorization | SVD (100 factors) | Highest accuracy |

---

## 📸 Screenshots

### API Documentation
![Swagger UI](screenshots/swagger-ui.png)

### Recommendations Response
![API Response](screenshots/api-response.png)

### Terminal Demo
![Terminal](screenshots/terminal-output.png)

### Data Visualizations
![Chart 1](screenshots/chart-1.png)
![Chart 2](screenshots/chart-2.png)

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python download_data.py

# Run demo
python main.py

# Start API
python api.py
