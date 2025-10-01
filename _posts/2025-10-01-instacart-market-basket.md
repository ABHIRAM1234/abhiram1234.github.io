---
layout: post
title: "Instacart Market Basket: Analysis, Modeling, and Real-Time Reorder API"
image: "/posts/instacart-title-img.png"
tags: [Recommendation Systems, Market Basket Analysis, Machine Learning, Python, XGBoost, Clustering, Association Rules, Deployment, Docker, GCP]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Problem & Data](#problem-data)
- [02. Exploratory Analysis & Customer Segmentation](#eda-segmentation)
- [03. Association Rules & Product Affinities](#association-rules)
- [04. Reorder Prediction Model](#reorder-model)
- [05. API Deployment (Docker + Cloud Run)](#deployment)
- [06. Results & Impact](#results)
- [07. Technical Stack](#tech-stack)
- [08. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

Unified two complementary Instacart projects into one end-to-end solution: exploratory analytics, segmentation, and association rules, followed by a production-grade reorder prediction model deployed as a real-time API.

---

## <a name="problem-data"></a>01. Problem & Data

- Goal: predict which previously purchased products a user will reorder next
- Dataset: multi-table relational data with user orders, products, and timestamps (millions of records)

---

## <a name="eda-segmentation"></a>02. Exploratory Analysis & Customer Segmentation

- Explored product popularity, weekly/hourly purchase patterns, and reorder tendencies
- Built customer segments using PCA + K-Means to capture shopping behaviors and frequency patterns

---

## <a name="association-rules"></a>03. Association Rules & Product Affinities

- Mined frequent itemsets and association rules (Apriori) to reveal co-purchase relationships
- Insights inform candidate generation and promotion bundling strategies

---

## <a name="reorder-model"></a>04. Reorder Prediction Model

- Engineered 30+ behavioral and temporal features (e.g., days since last order, user/product frequencies)
- Tuned XGBoost classifier achieving strong F1 performance on validation
- Dynamic Top-N strategy personalizes number of recommended items per user

---

## <a name="deployment"></a>05. API Deployment (Docker + Cloud Run)

- Packaged as a Flask + Gunicorn service, containerized via Docker
- Deployed to Google Cloud Run for scalable, low-ops serving
- Feature store backed by BigQuery for fast lookups at inference time

Local test example (after running the container):
```
POST http://127.0.0.1:5000/predict
Body: {"user_id": 1}
```

---

## <a name="results"></a>06. Results & Impact

- Accurate reorder predictions enable targeted recommendations and basket growth
- Customer segments and rules guide merchandising and personalized campaigns

---

## <a name="tech-stack"></a>07. Technical Stack

- Python, Pandas, Scikit-learn, XGBoost, Optuna
- Notebooks for EDA/feature engineering; Flask, Gunicorn, Docker for serving
- Google Cloud Run + BigQuery for scalable inference

---

## <a name="links"></a>08. Project Links

- **Reorder API + Modeling**: [Instacart-Reorder-Prediction](https://github.com/ABHIRAM1234/Instacart-Reorder-Prediction)
- **EDA, Segmentation, Rules**: [instacart-orders](https://github.com/ABHIRAM1234/instacart-orders)

---

## 🚀 Why This Project Matters

Combines analytical depth (segmentation and rules) with production engineering (real-time API) to demonstrate a full lifecycle retail ML system—from insights to impact.


