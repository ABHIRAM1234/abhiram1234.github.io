---
layout: post
title: "Instacart Reorder Prediction: An End-to-End MLOps Project"
image: "/posts/instacart-title-image.png"
tags: [Machine Learning, Classification, XGBoost, MLOps, Feature Engineering, Deployment, Google Cloud]
---

This project documents the complete lifecycle of solving the Instacart Market Basket Analysis challenge. The goal was to build a system that accurately predicts which products a user will reorder. This report covers every step, from initial data exploration and advanced feature engineering to model optimization and final deployment as a live API on Google Cloud.

# Table of Contents
- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth & Next Steps](#overview-growth)
- [01. Data Understanding & EDA](#data-overview)
- [02. The Methodology: Binary Classification](#methodology)
- [03. Feature Engineering: The Core of the Solution](#feature-engineering)
    - [Foundational Features](#features-foundational)
    - [Challenge 1: Advanced Temporal Features](#features-temporal)
    - [Challenge 2: The Prod2Vec Experiment](#features-p2v)
- [04. Modeling & Optimization](#modeling)
    - [Model Selection](#modeling-selection)
    - [Hyperparameter Tuning with Optuna](#modeling-tuning)
    - [Post-Processing: Dynamic Top-N Predictions](#modeling-post)
- [05. Deployment: From Notebook to Live API](#deployment)
    - [Challenge 3: The Memory Limit Saga](#deployment-challenge)
    - [The Final Architecture: A Scalable Solution](#deployment-architecture)
- [06. Conclusion & Next Steps](#conclusion)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

For an e-commerce grocery platform like Instacart, a significant portion of business comes from repeat purchases. A key challenge is to enhance user convenience and drive engagement by accurately predicting what a customer will reorder. This project's objective was to build a machine learning system to tackle this prediction task.

### Actions <a name="overview-actions"></a>

The project was framed as a large-scale **binary classification** problem: for every product a user has purchased previously, will they reorder it in their next basket?

The end-to-end workflow involved:
1.  **In-depth EDA** of a relational dataset containing over 3 million orders.
2.  **Extensive Feature Engineering** to create a rich dataset capturing user habits, product popularity, and user-product interaction history. This included a crucial pivot to high-resolution temporal features.
3.  **Model Optimization**, where an `XGBoost` model was systematically tuned using the `Optuna` framework to maximize the F1 Score.
4.  **Containerization & Deployment**, where the entire prediction pipeline was packaged with `Docker` and deployed as a live, serverless API on **Google Cloud Run**, backed by **Google BigQuery** as a feature store.

### Results <a name="overview-results"></a>

The final solution is a robust and scalable prediction system.
-   The champion model is a tuned **XGBoost Classifier** that achieved an **F1 Score of 0.4506** on a held-out validation set.
-   The deployment architecture successfully handles real-world resource constraints by using BigQuery for heavy feature lookups, allowing the API to run on a lean **1GiB of memory**.
-   The live API endpoint can serve real-time predictions for any given user ID.

### Growth/Next Steps <a name="overview-growth"></a>

While the current system is highly effective, future iterations could explore:
-   **Predictive "N":** Building a secondary regression model to predict the *exact number* of items a user will reorder, making the Top-N strategy even more dynamic.
-   **Model Stacking:** Ensembling the XGBoost model with a fundamentally different model (like a Factorization Machine or Neural Network) to capture a more diverse set of patterns.
-   **Feature Store Optimization:** Downcasting data types (`float64` -> `float32`, etc.) in the BigQuery feature store to reduce storage costs and query processing time even further.

___

# 01. Data Understanding & EDA <a name="data-overview"></a>

The dataset consisted of six CSV files, representing a relational database. The core challenge was to merge these tables to create a unified view of each user's purchase history.

-   **`orders.csv`**: The central table containing `user_id`, `order_id`, and order timing.
-   **`order_products__*.csv`**: Linking tables showing which products were in which orders.
-   **`products.csv`**: A lookup table for product names.

A key finding from EDA was the diversity of user behavior—some shopped weekly, others monthly; some bought few items, others bought many. This immediately highlighted that a one-size-fits-all approach would be ineffective and that user-specific features were critical.

___

# 02. The Methodology: Binary Classification <a name="methodology"></a>

At its heart, this is a **binary classification** problem. For every candidate pair of `(user_id, product_id)`, we predict one of two outcomes:
-   **Class 1:** The user reorders the product.
-   **Class 0:** The user does not reorder the product.

#### Model Choice: XGBoost
An `XGBoost Classifier` was chosen for its proven, state-of-the-art performance on structured, tabular data. It is highly efficient and capable of capturing complex, non-linear relationships in the features.

#### Evaluation Metric: F1 Score
The dataset is naturally imbalanced—a user typically reorders only a small fraction of all products they've ever bought. Therefore, simple accuracy is a poor metric. The **F1 Score** was chosen as the primary metric because it provides a harmonic mean of precision and recall, offering a robust measure of model performance on imbalanced classes.

___

# 03. Feature Engineering: The Core of the Solution <a name="feature-engineering"></a>

This was the most critical and iterative part of the project.

### Foundational Features <a name="features-foundational"></a>
A comprehensive set of over 30 features was initially engineered, grouped into three categories:
1.  **User Features:** `user_total_orders`, `user_reorder_ratio`, `user_avg_basket_size`, `user_avg_days_since_prior`.
2.  **Product Features:** `product_reorder_rate`, `product_unique_users`.
3.  **User-Product Features:** `up_total_purchases`, `up_orders_since_last_purchase`, `up_reorder_rate`.

### Challenge 1: Advanced Temporal Features <a name="features-temporal"></a>
-   **The Problem:** The initial feature `up_orders_since_last_purchase` was good but imprecise. A "gap of 1 order" could be two days or two months.
-   **The Hypothesis:** Measuring time in precise **days** would provide a much stronger signal to the model.
-   **The Solution:** A user-specific timeline was built by calculating the cumulative sum of `days_since_prior_order`. This enabled the creation of new, more powerful features:
    -   **`days_since_product_last_purchased`**: The exact number of days since a user last bought an item.
    -   **`user_product_avg_purchase_gap`**: The average time in days between a user's purchases of a specific product.
-   **The Result:** Adding these features and re-training the tuned model resulted in a direct F1 score improvement from **0.4497** to **0.4506**. This was a major success, validating the hypothesis and demonstrating the power of iterative feature improvement.

### Challenge 2: The Prod2Vec Experiment <a name="features-p2v"></a>
-   **The Hypothesis:** The model has no semantic understanding of products (e.g., that "Avocado" and "Organic Avocado" are similar). Training product embeddings with `Word2Vec` could capture these relationships.
-   **The Action:** A `Word2Vec` model was trained on order histories, creating a 20-dimensional vector for each product.
-   **The Result:** When these new features were added, the model's F1 score unexpectedly *dropped*. This was a valuable negative result, indicating that the hyperparameters tuned for the original feature set were not suitable for this new, wider dataset. The decision was made to discard these features to avoid unnecessary complexity and stick with the proven champion model.

___

# 04. Modeling & Optimization <a name="modeling"></a>

### Model Selection <a name="modeling-selection"></a>
The `XGBoost Classifier` consistently performed well and was selected as the final model for optimization and deployment.

### Hyperparameter Tuning with Optuna <a name="modeling-tuning"></a>
-   **The Challenge:** The default XGBoost parameters are a starting point, not an optimized solution.
-   **The Action:** A 50-trial study was conducted using the **Optuna** framework to automatically search for the best hyperparameters, with the objective of maximizing the F1 score on the validation set.
-   **The Result:** The tuning process discovered a new set of parameters that significantly improved the F1 score from **~0.444** to **0.4497**.

### Post-Processing: Dynamic Top-N Predictions <a name="modeling-post"></a>
-   **The Insight:** A single probability threshold (e.g., > 0.22) treats all users the same, which is suboptimal.
-   **The Solution:** A dynamic **"Top-N"** strategy was implemented. For each user, we first calculate their historical average number of reordered items, `N`. The final prediction then consists of the `N` products with the highest predicted probabilities for that user. This personalizes the size of the predicted basket, making the recommendations more relevant.

___

# 05. Deployment: From Notebook to Live API <a name="deployment"></a>

This phase focused on turning the successful model into a real-world application, encountering and solving classic MLOps challenges.

### Challenge 3: The Memory Limit Saga <a name="deployment-challenge"></a>
-   **The Problem:** When the containerized application was deployed to Google Cloud Run, it repeatedly crashed due to memory limits, even after increasing the allocated RAM from 512MiB to 4GiB. The logs showed that loading the entire `feature_store.parquet` file at startup consumed over 4.6 GiB of RAM.
-   **The "Aha!" Moment:** The architectural pattern was flawed. A serverless API should not load a multi-gigabyte feature store into memory. This is slow, expensive, and doesn't scale.

### The Final Architecture: A Scalable Solution <a name="deployment-architecture"></a>
The system was re-architected to be robust and efficient, following professional MLOps best practices.
1.  **Offline Data Warehouse:** The `feature_store.parquet` file was uploaded to **Google BigQuery**, which acts as a powerful, serverless feature store.
2.  **Lightweight API:** The Flask application was rewritten to be stateless and lightweight. On receiving a request, it **queries** BigQuery for only the features of the specific `user_id`, rather than loading the entire table.
3.  **Successful Deployment:** This new architecture deployed successfully on Google Cloud Run with only **1GiB of RAM**. It is faster to start up, more cost-effective, and can scale to handle a massive number of users without memory issues.

The final stack consists of:
-   **Backend:** Flask (API), Gunicorn (WSGI Server)
-   **Containerization:** Docker
-   **Cloud Platform:** Google Cloud Run (Serverless Compute), Google BigQuery (Feature Store)

___

# 06. Conclusion & Next Steps <a name="conclusion"></a>

This project successfully delivered a high-performing, production-ready product reorder prediction system. The journey from raw data to a live cloud API highlighted the critical importance of iterative feature engineering, systematic model optimization, and—most crucially—designing scalable deployment architectures that solve real-world resource constraints. The final system is not just an accurate model, but a well-engineered and robust machine learning product.