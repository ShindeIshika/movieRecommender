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
Then open: http://localhost:8000/docs

📁 Files
File	What it does
main.py	Run all 3 approaches
api.py	REST API server
content_based.py	Genre-based recommendations
collaborative.py	User-user collaborative filtering
matrix_factorization.py	SVD model
evaluate_all.py	Precision/Recall metrics
📈 Sample Output
text
Recommendations for User 1 (SVD):
  - Shawshank Redemption (4.89)
  - Seven Samurai (4.88)
  - Usual Suspects (4.77)
🛠️ Built With
Python, Pandas, NumPy

Scikit-learn, Surprise

FastAPI, Uvicorn
