---
layout: post
title: "Instacart Market Basket: Analysis, Modeling, and Real-Time Reorder API"
image: "/posts/instacart-title-img.png"
tags: [Recommendation Systems, Market Basket Analysis, Machine Learning, Python, XGBoost, Clustering, Association Rules, Deployment, Docker, GCP]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Why This Problem Matters](#why-this-matters)
- [02. Key Concepts](#concepts)
- [03. My Step-by-Step Thought Process](#thought-process)
- [04. Problem & Data](#problem-data)
- [05. Exploratory Analysis & Customer Segmentation](#eda-segmentation)
- [06. Association Rules & Product Affinities](#association-rules)
- [07. Reorder Prediction Model](#reorder-model)
- [08. API Deployment (Docker + Cloud Run)](#deployment)
- [09. Results & Impact](#results)
- [10. Technical Stack](#tech-stack)
- [11. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

**What this project is (in plain English):**  
I built an end-to-end system that **predicts which products a customer will reorder** on their next Instacart order. The business goal is to grow basket size and retention by showing "reorder" suggestions that are relevant (precision) and capture as many actual reorders as possible (recall). I did not try to predict *all* products—that would be too many. Instead I (1) **explored the data** to understand who buys what and how often, (2) **segmented customers** (e.g. weekly shoppers vs bulk buyers) for insights and as model features, (3) **mined association rules** (Market Basket Analysis) to find co-purchase patterns and to **narrow the prediction space** (candidate generation), and (4) **trained an XGBoost model** to rank candidates and output a personalized Top-N reorder list per user. Finally I **deployed the model as a real-time API** (Flask in Docker on Google Cloud Run) so it can be called with a `user_id` and return ranked product recommendations.

Unified two complementary Instacart projects into one end-to-end solution: exploratory analytics, segmentation, and association rules, followed by a production-grade reorder prediction model deployed as a real-time API.

**At a glance**
- **Objective:** Predict next-order reorders to grow basket size and retention.
- **Approach:** MBA + Segmentation for candidates and insights; XGBoost ranker for final predictions.
- **Serving:** Flask API on Cloud Run; BigQuery-backed features; Docker for reproducibility.
- **Code:** [Instacart-Reorder-Prediction](https://github.com/ABHIRAM1234/Instacart-Reorder-Prediction) (API + modeling) | [instacart-orders](https://github.com/ABHIRAM1234/instacart-orders) (EDA, segmentation, rules).
- **Highlights:** Interview-ready narrative with code, diagrams, and deployment steps.

How I’d summarize this in an interview: I framed the business problem around increasing reorder rate and basket size. I cleaned and modeled the Instacart dataset to understand customer behavior, used MBA and segmentation for insights and candidate generation, and built an XGBoost-based reorder model exposed via a low-latency API on Cloud Run.

---

## <a name="why-this-matters"></a>01. Why This Problem Matters

**Business context:** Grocery and delivery platforms (e.g. Instacart) make money when customers order more often and add more items per order. "Reorder" suggestions are a direct lever: if we predict reorders accurately, we can surface the right products and grow basket size without annoying users with irrelevant suggestions.

**What we need:** (1) **Precision:** Top recommendations should be items the user actually reorders. (2) **Recall:** We want to capture as many true reorders as possible in the Top-N. (3) **Scalability:** We need to **limit candidates** per user (e.g. products they've bought before, or high-lift co-purchases) and then **rank** those candidates.

**Why this approach:** I used **MBA** and **segmentation** for **insights** and **candidate generation**, then a single **XGBoost ranker** to produce the final Top-N. Deploying as an **API** makes it usable by other services and demonstrates production engineering.

---

## <a name="concepts"></a>02. Key Concepts: Reorder Prediction, MBA, and Candidate Generation

**Reorder prediction:** For each user, we predict which **previously purchased** products they will buy again in their next order. The label is binary; we optimize for F1 on a held-out set.

**Market Basket Analysis (MBA):** We mine **frequent itemsets** and **association rules** (support, confidence, lift). This gives (1) **insights** for merchandising and (2) **candidates**: for a user who bought eggs and bread, we bias the candidate set toward products that co-occur with those before we rank.

**Candidate generation:** We **restrict** the set of products per user (e.g. only products they've bought before, optionally + MBA affinities). The **ranker** (XGBoost) scores only those candidates and we return the Top-N.

**Segmentation (PCA + K-Means):** We reduce user behavior with **PCA**, then **cluster** users (e.g. weekly shoppers, bulk buyers). Segments serve as **marketing personas** and **model features**.

---

## <a name="thought-process"></a>03. My Step-by-Step Thought Process

**Step 1: Define the business metric and the prediction task** — I started from the goal: increase reorder rate and basket size. The evaluation is F1 on the "train" split. I decided to **limit the prediction space** to candidates (e.g. previously purchased + MBA affinities) and then **rank** those with a single model.

**Step 2: Explore the data and build customer segments** — I loaded orders, order_products, and products; explored product popularity, weekly/hourly patterns, and reorder rates; built user-level features and used **PCA + K-Means** to segment users. I did this for (1) **insights** and (2) **features** for the ranker.

**Step 3: Mine association rules and use them for candidate generation** — I ran **Apriori** to find frequent itemsets and association rules. I used these for (1) **insights** and (2) **candidate generation**: for each user, besides "products they've bought before," I could add products with high lift given their recent baskets.

**Step 4: Engineer features and train the ranker** — For each user–candidate pair I built behavioral and temporal features (order rate, orders since last purchase, add-to-cart order, user stats, segment/MBA features). I labeled each pair and trained an **XGBoost** classifier, tuned the threshold for F1, and used a **dynamic Top-N** per user.

**Step 5: Deploy as a real-time API** — I packaged the model and feature logic in a **Flask** app, containerized with **Docker**, and deployed to **Google Cloud Run**. The API accepts `user_id` and returns a ranked Top-N list.

---

## <a name="problem-data"></a>04. Problem & Data

- **Goal:** Predict which previously purchased products a user will reorder next
- Dataset: multi-table relational data with user orders, products, and timestamps (millions of records)

Interview angle: The KPI was F1 on the held-out “train” split emulating Kaggle’s evaluation. The operational goals were precision for top-N recommendations (reduce irrelevant suggestions) and recall for basket growth.

---

## <a name="eda-segmentation"></a>05. Exploratory Analysis & Customer Segmentation

- Explored product popularity, weekly/hourly purchase patterns, and reorder tendencies
- Built customer segments using PCA + K-Means to capture shopping behaviors and frequency patterns

Interview angle: Segmentation served two purposes—1) marketing personas for campaigns and 2) model features that encode user intent. I reduced sparsity with PCA, then clustered to capture cadence (weekly shoppers vs bulk buyers) and diversity (broad vs niche baskets).

---

## <a name="association-rules"></a>06. Association Rules & Product Affinities

- Mined frequent itemsets and association rules (Apriori) to reveal co-purchase relationships
- Insights inform candidate generation and promotion bundling strategies

Interview angle: MBA was used both for insights and as a candidate generator to limit the prediction space per user. For example, given eggs and bread, peanut butter and jam have high lift; we bias candidates toward those affinities before ranking.

---

## <a name="reorder-model"></a>07. Reorder Prediction Model

- Engineered 30+ behavioral and temporal features (e.g., days since last order, user/product frequencies)
- Tuned XGBoost classifier achieving strong F1 performance on validation
- Dynamic Top-N strategy personalizes number of recommended items per user

Interview angle: I combined three strands—segmentation features, MBA-based candidate generation, and user–product interaction features—into a single ranking model (XGBoost). I tuned the classification threshold to optimize F1 and applied a dynamic Top‑N per user to balance precision and recall.

```python
# Feature engineering (simplified)
import pandas as pd
import numpy as np

orders = pd.read_csv("orders.csv")
order_products = pd.read_csv("order_products__prior.csv")
products = pd.read_csv("products.csv")

user_orders = orders[orders["eval_set"] == "prior"][["order_id","user_id","order_number","days_since_prior_order"]]
op = order_products.merge(user_orders, on="order_id", how="left")

# Per-user stats
user_feat = op.groupby("user_id").agg(
    user_num_orders=("order_number","max"),
    user_total_products=("product_id","count"),
    user_distinct_products=("product_id","nunique"),
    user_avg_days_between=("days_since_prior_order","mean")
).reset_index()

# User-product interaction features
up = op.groupby(["user_id","product_id"]).agg(
    up_orders=("order_id","nunique"),
    up_last_order=("order_number","max"),
    up_first_order=("order_number","min"),
    up_cart_mean=("add_to_cart_order","mean")
).reset_index()
up = up.merge(user_feat[["user_id","user_num_orders"]], on="user_id")
up["up_order_rate"] = up["up_orders"] / up["user_num_orders"].clip(lower=1)
up["up_orders_since_last"] = up["user_num_orders"] - up["up_last_order"]

# Label for training on the "train" eval set
train_orders = orders[orders["eval_set"] == "train"]["order_id"]
train_truth = pd.read_csv("order_products__train.csv")[["order_id","product_id"]]
train_truth["label"] = 1

candidates = up.merge(orders[orders["eval_set"] == "train"]["user_id"].to_frame().merge(train_orders.to_frame(), left_index=True, right_index=True, how="left"), on="user_id")
candidates = candidates.merge(products[["product_id"]], on="product_id", how="inner")
X = candidates.merge(train_truth, on=["order_id","product_id"], how="left").fillna({"label":0})

feature_cols = [
    "up_order_rate","up_orders_since_last","up_cart_mean",
    "user_num_orders","user_total_products","user_distinct_products","user_avg_days_between"
]

# Train XGBoost with Optuna
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

X_train, X_val, y_train, y_val = train_test_split(
    X[feature_cols], X["label"].astype(int), test_size=0.2, random_state=42, stratify=X["label"]
)

def objective(trial):
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": trial.suggest_float("eta", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 10.0)
    }
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dval, "val")], early_stopping_rounds=100, verbose_eval=False)
    preds = (model.predict(dval) > 0.21).astype(int)  # tuned threshold
    return 1.0 - f1_score(y_val, preds)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_params
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
model = xgb.train({**best_params, "objective":"binary:logistic", "eval_metric":"logloss"}, dtrain,
                  num_boost_round=2000, evals=[(dval,"val")], early_stopping_rounds=100, verbose_eval=False)

# Dynamic Top-N selection (sketch)
user_thresholds = X.groupby("user_id")["label"].mean().clip(lower=0.1, upper=0.6)
```
---

## <a name="deployment"></a>08. API Deployment (Docker + Cloud Run)

- Packaged as a Flask + Gunicorn service, containerized via Docker
- Deployed to Google Cloud Run for scalable, low-ops serving
- Feature store backed by BigQuery for fast lookups at inference time

Interview angle: I containerized a Flask API with Gunicorn, stored model artifacts in GCS, and served via Cloud Run with autoscaling. BigQuery provides low-latency feature fetches, and the interface returns a ranked Top‑N list per user. Local Docker parity ensures reproducibility.

Local test example (after running the container):
```
POST http://127.0.0.1:5000/predict
Body: {"user_id": 1}
```

Sample response

```json
{
  "user_id": 1,
  "products": [196, 493, 24852, 13176, 27845, 21137, 47209, 16797, 21903, 47626],
  "scores": [0.91, 0.88, 0.86, 0.84, 0.82, 0.80, 0.79, 0.77, 0.76, 0.75]
}
```

Architecture

![Instacart API Architecture](/img/posts/instacart-reorder-architecture.svg)

<sub>Figure: Minimal real-time API on Cloud Run with Docker, pulling features from BigQuery and model artifacts from GCS (or local volume in dev).</sub>



Minimal Flask API (deployment/app.py)

```python
from flask import Flask, request, jsonify
import xgboost as xgb
import pandas as pd
import os

app = Flask(__name__)

# Load model/artifacts at startup
MODEL_PATH = os.getenv("MODEL_PATH", "/app/artifacts/model.bst")
FEATURES = ["up_order_rate","up_orders_since_last","up_cart_mean","user_num_orders","user_total_products","user_distinct_products","user_avg_days_between"]
bst = xgb.Booster()
bst.load_model(MODEL_PATH)

def fetch_user_features(user_id: int) -> pd.DataFrame:
    # Replace with BigQuery call in prod; here assume local CSV cache
    df = pd.read_csv("/app/artifacts/user_features.csv")
    return df[df["user_id"] == user_id][FEATURES]

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    user_id = int(payload["user_id"])
    feats = fetch_user_features(user_id)
    if feats.empty:
        return jsonify({"user_id": user_id, "products": []})
    dm = xgb.DMatrix(feats)
    scores = bst.predict(dm)
    # Return top-N based on score threshold or fixed N
    top_n = int(payload.get("top_n", 10))
    # In practice join with candidate product ids; placeholder ids 1..N
    return jsonify({"user_id": user_id, "products": list(range(1, top_n+1)), "scores": scores[:top_n].tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

Dockerfile (deployment/Dockerfile)

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY deployment/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY deployment/app.py /app/app.py
COPY artifacts/ /app/artifacts/
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run locally

```bash
docker build -t instacart-recommender -f deployment/Dockerfile .
docker run -d -p 5000:5000 -e MODEL_PATH=/app/artifacts/model.bst instacart-recommender
```

Deploy to Cloud Run

```bash
gcloud builds submit --tag gcr.io/$GOOGLE_CLOUD_PROJECT/instacart-recommender
gcloud run deploy instacart-recommender \
  --image gcr.io/$GOOGLE_CLOUD_PROJECT/instacart-recommender \
  --platform managed --region us-central1 --allow-unauthenticated
```

---

## <a name="results"></a>09. Results & Impact

- Accurate reorder predictions enable targeted recommendations and basket growth
- Customer segments and rules guide merchandising and personalized campaigns

---

## <a name="tech-stack"></a>10. Technical Stack

- Python, Pandas, Scikit-learn, XGBoost, Optuna
- Notebooks for EDA/feature engineering; Flask, Gunicorn, Docker for serving
- Google Cloud Run + BigQuery for scalable inference

---

## <a name="links"></a>11. Project Links

- **Reorder API + Modeling**: [Instacart-Reorder-Prediction](https://github.com/ABHIRAM1234/Instacart-Reorder-Prediction)
- **EDA, Segmentation, Rules**: [instacart-orders](https://github.com/ABHIRAM1234/instacart-orders)

---

## 🚀 Why This Project Matters

Combines analytical depth (segmentation and rules) with production engineering (real-time API) to demonstrate a full lifecycle retail ML system—from insights to impact.


