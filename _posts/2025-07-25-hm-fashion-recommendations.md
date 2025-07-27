---
layout: post
title: "H&M Personalized Fashion Recommendations: Iterative, Research-Driven Silver Medal Solution"
image: "/img/posts/h&m-rec-sys-title-img.png"
tags: [Recommender Systems, Machine Learning, Collaborative Filtering, Kaggle, Python, LightGBM, Deep Learning, Ensemble]
---

## Project Overview

In the H&M Personalized Fashion Recommendations Kaggle competition, I set out to build a state-of-the-art recommendation system for a global fashion retailer. The challenge: predict which items each customer would purchase next, using a massive dataset of 31M+ transactions, 1M+ customers, and 100K+ products. My solution earned a **Silver Medal (Top 2%)**, ranking 45th out of 3,006 teams.

[GitHub Repository](https://github.com/ABHIRAM1234/H-M-Fashion-Recommendations)

---

## My Thought Process & Iterative Approach

### 1. Understanding the Problem & Data (Exploratory Phase)
I began with deep exploratory data analysis, inspired by [Vanguarde’s EDA](https://www.kaggle.com/code/vanguarde/h-m-eda-first-look) and [Gpreda’s EDA](https://www.kaggle.com/code/gpreda/h-m-eda-and-prediction). I asked: What drives customer purchases? How do product attributes, seasonality, and demographics interact? I discovered strong seasonality, product popularity trends, and diverse customer behaviors.

### 2. Building a Baseline (First Models)
My initial models were simple popularity- and recency-based recommenders. These provided a baseline and validated my data pipeline, but lacked personalization and nuance.

### 3. Feature Engineering & Candidate Generation
To improve, I engineered features such as recency, frequency, monetary value, product attributes, and customer demographics. I then implemented multiple candidate generation strategies:
- **Collaborative Filtering:** Using Implicit ALS ([julian3833’s notebook](https://www.kaggle.com/code/julian3833/h-m-implicit-als-model-0-014)), I generated personalized recommendations based on user-item interaction patterns.
- **Co-occurrence & Association Rules:** Leveraged [cdeotte’s co-purchase approach](https://www.kaggle.com/code/cdeotte/recommend-items-purchased-together-0-021) to find items frequently bought together.
- **Heuristics & Rules:** Added time-based and segment-based rules for cold-start users.

This multi-pronged approach ensured both global trends and individual preferences were captured.

### 4. Model Development & Iterative Improvement
With diverse candidates, I moved to ranking models:
- **LightGBM Ranker**
- **LightGBM Classifier**
- **Deep Neural Network (DNN)**

I validated each model using time-based splits ([bearcater’s notebook](https://www.kaggle.com/code/bearcater/h-m-personalized-fashion-recommendations)) and tracked leaderboard scores. No single model was sufficient—each had unique strengths.

### 5. Ensembling & Robustness
To maximize performance, I adopted an ensemble approach:
- Blended multiple recall strategies ([tarique7’s multi-blend](https://www.kaggle.com/code/tarique7/lb-0-0240-h-m-ensemble-magic-multi-blend), [aruaru0’s ensemble](https://www.kaggle.com/code/aruaru0/h-and-m-ensamble-only-dadfc6)).
- Stacked outputs from different ranking models ([jaloeffe92’s ensemble](https://www.kaggle.com/code/jaloeffe92/lb-0-0236-ensemble-gives-you-bronze-medal)).
- Used weighted averaging to combine predictions and reduce overfitting.

This ensemble approach consistently outperformed any single model and improved robustness.

### 6. Resource Management & Scalability
Given hardware constraints (50GB RAM), I engineered an efficient pipeline:
- Limited training to the most recent 4 weeks of data.
- Generated ~50 candidates per user to balance recall and computational cost.
- Used intermediate data storage and batch processing to avoid memory bottlenecks.

---

## Key Results & Impact
- **Leaderboard:** Silver Medal (Top 2%), Rank 45/3006.
- **Model Performance:** Public LB 0.0292, Private LB 0.02996.
- **Business Value:** Demonstrated scalable, production-ready recommendation techniques for real-world e-commerce.

---

## Technical Stack
- **Languages:** Python, Jupyter Notebook
- **Libraries:** LightGBM, PyTorch/TensorFlow (DNN), Pandas, NumPy, Scikit-learn
- **Tools:** Jupyter, Sphinx (docs), GitHub Actions (CI)
- **Workflow:** Cookiecutter Data Science template

---

## What I Learned
- **Iterative Experimentation:** Each step—EDA, feature engineering, model building, ensembling—brought new insights and improvements.
- **Diversity Wins:** Combining different models and strategies is more powerful than hyper-tuning a single approach.
- **Resourcefulness:** Efficient engineering is crucial when working with large-scale data and limited hardware.
- **Collaboration & Research:** Leveraged and built upon the best ideas from the Kaggle community, adapting them to my own workflow and constraints.

---

## References & Influences
- [Kaggle Competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- [Vanguarde EDA](https://www.kaggle.com/code/vanguarde/h-m-eda-first-look)
- [cdeotte Co-purchase](https://www.kaggle.com/code/cdeotte/recommend-items-purchased-together-0-021)
- [julian3833 ALS Model](https://www.kaggle.com/code/julian3833/h-m-implicit-als-model-0-014)
- [gpreda EDA & Prediction](https://www.kaggle.com/code/gpreda/h-m-eda-and-prediction)
- [tarique7 Multi-blend](https://www.kaggle.com/code/tarique7/lb-0-0240-h-m-ensemble-magic-multi-blend)
- [jaloeffe92 Ensemble](https://www.kaggle.com/code/jaloeffe92/lb-0-0236-ensemble-gives-you-bronze-medal)
- [bearcater Final Notebook](https://www.kaggle.com/code/bearcater/h-m-personalized-fashion-recommendations)
- [aruaru0 Ensemble](https://www.kaggle.com/code/aruaru0/h-and-m-ensamble-only-dadfc6)
- [fanot89 Cringe](https://www.kaggle.com/code/fanot89/hm-cringe)
- [jacob34 Final Shared](https://www.kaggle.com/code/jacob34/clear-n-simple-final-shared-notebook)

---

This project is a testament to my ability to tackle large-scale, real-world data science problems with a structured, iterative, and research-driven approach. 