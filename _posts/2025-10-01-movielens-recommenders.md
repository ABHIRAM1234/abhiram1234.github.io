---
layout: post
title: "MovieLens Recommenders: Content-Based, CF, SVD, and Deep Learning"
image: "/posts/recommendation-title-img.png"
tags: [Recommender Systems, Machine Learning, Python, Jupyter, SVD, Deep Learning, Collaborative Filtering]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Dataset & Problem](#dataset)
- [02. Models Implemented](#models)
- [03. Pipeline & Notebooks](#pipeline)
- [04. Results & Learnings](#results)
- [05. Requirements](#requirements)
- [06. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

Built and compared four recommendation approaches on MovieLens 1M: content-based filtering, user-item collaborative filtering, matrix factorization (SVD), and a deep learning model. The goal is to understand trade-offs in accuracy, scalability, and interpretability.

---

## <a name="dataset"></a>01. Dataset & Problem

- MovieLens 1M: 1,000,209 ratings, ~3,900 movies, 6,040 users
- Task: predict user preferences and recommend top-N movies
- Inputs: `users.csv`, `movies.csv`, `ratings.csv`

---

## <a name="models"></a>02. Models Implemented

- Content-Based: movie metadata similarity
- Memory-based CF: user-user and item-item similarities
- SVD: latent factor model via matrix factorization
- Deep Learning: embedding-based model for user/movie interactions

Code references: `CFModel.py`, notebooks for each approach.

---

## <a name="pipeline"></a>03. Pipeline & Notebooks

- `Data_Processing.ipynb`: load/clean/feature prep
- `Content_Based_and_Collaborative_Filtering_Models.ipynb`
- `SVD_Model.ipynb`
- `Deep_Learning_Model.ipynb`

---

## <a name="results"></a>04. Results & Learnings

- SVD provides strong baseline with good accuracy/efficiency
- Deep model learns nuanced interactions but needs tuning and compute
- Content-based is interpretable and cold-start friendly for items

---

## <a name="requirements"></a>05. Requirements

- Python 3.x, Jupyter
- Key libs: pandas, numpy, scipy, scikit-learn, surprise, keras, h5py, matplotlib, seaborn

---

## <a name="links"></a>06. Project Links

- **GitHub Repository**: [movielens](https://github.com/ABHIRAM1234/movielens)
- **Dataset**: [MovieLens](https://grouplens.org/datasets/movielens/)

---

## 🚀 Why This Project Matters

Demonstrates practical recommender system implementations and comparison across classical and neural approaches—skills widely applicable to personalization and ranking problems.


