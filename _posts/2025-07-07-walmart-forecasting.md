---
layout: post
title: "M5 Forecasting: A Strategic Approach to Large-Scale Time-Series Forecasting"
image: "img/posts/m5-forecasting-title-image.png"
tags: [Machine Learning, Time Series, Forecasting, LightGBM, Python, Pandas, MLOps]
---

# Table of Contents
- [00. Project Overview](#overview-main)
    - [The Business Problem](#overview-context)
    - [My Strategic Approach](#overview-actions)
    - [The Outcome](#overview-results)
- [01. Data Engineering: Building a Foundation for Scale](#data-overview)
- [02. The Core Strategy: Direct Forecasting for Stability](#methodology)
- [03. Feature Engineering: Giving the Model a Memory](#feature-engineering)
- [04. Modeling: A Multi-Stage Optimization Strategy](#modeling)
    - [Stage 1: The Robust LightGBM Baseline](#modeling-champion1)
    - [Stage 2: Advanced Optimization with "Dark Magic"](#modeling-champion2)
    - [Stage 3: Automated Hyperparameter Tuning](#modeling-champion3)
- [05. Evaluation & Results: Measuring Success](#prediction)
- [06. Conclusion & Key Learnings](#conclusion)

___

<a name="overview-main"></a>
## 00. Project Overview

This project deconstructs the M5 forecasting challenge, one of the most comprehensive and large-scale time-series competitions ever held. It demonstrates an end-to-end process for developing a state-of-the-art model capable of generating accurate, long-term forecasts for tens of thousands of individual products.

<a name="overview-context"></a>
### The Business Problem

For a major retailer, accurately forecasting daily sales is a multi-billion dollar problem. Inefficiencies lead to two critical business costs: **stock-outs**, resulting in lost sales and poor customer satisfaction, and **overstock**, which ties up capital and shelf space in non-performing inventory. The goal was to build a system to forecast daily demand for over 30,000 unique Walmart products across 10 stores for the next 28 days.

<a name="overview-actions"></a>
### My Strategic Approach

I adopted a multi-stage approach focused on building a single, powerful LightGBM model. The strategy prioritized robust data engineering, extensive feature creation, and iterative model optimization. This involved:
1.  **Engineering a scalable data pipeline** capable of handling over 46 million rows of data efficiently.
2.  **Creating hundreds of predictive features** to capture seasonality, price effects, and special events.
3.  **Iteratively training and validating a LightGBM model**, starting with a simple baseline and progressively adding complexity to maximize predictive accuracy.

<a name="overview-results"></a>
### The Outcome

The final optimized model achieved a **Weighted Root Mean Squared Scaled Error (WRMSSE) of 0.51**, a score that would be highly competitive and place firmly in the top tier of the competition. This result validated that a single, well-engineered gradient boosting model can deliver state-of-the-art performance on a massive and complex forecasting task.

___

<a name="data-overview"></a>
## 01. Data Engineering: Building a Foundation for Scale

The foundation of any successful ML project is a robust data pipeline. The M5 dataset, once transformed for modeling, expanded to over 46 million rows, making memory management a critical first step.

My data engineering process involved:
*   **Data Transformation:** Melting the wide-format sales data into a long format suitable for machine learning, where each row represents a single product on a single day.
*   **Aggressive Memory Optimization:** I implemented a script to **downcast** all numerical data types to their most efficient size (e.g., `int64` to `int16`, `float64` to `float32`). This single step reduced the memory footprint by over 70%, making it possible to process the entire dataset on standard hardware.
*   **Target Normalization:** To stabilize model training, I applied a `log1p` transformation to the sales target. This compresses the wide range of sales values and helps the model equally weigh errors on both low-selling and high-selling items.

___

<a name="methodology"></a>
## 02. The Core Strategy: Direct Forecasting for Stability

Forecasting 28 days into the future presents a key strategic choice: do you predict one day at a time recursively, or predict all 28 days at once?

I chose the **direct forecasting** strategy. This involves training a single model to predict all 28 future days simultaneously as 28 distinct output targets.

**Why this approach?**
1.  **Stability:** Recursive forecasting, where each day's prediction is used as a feature for the next day's prediction, can suffer from **error accumulation**. A mistake on Day 1 can cascade and worsen through Day 28. The direct approach is more stable as each of the 28 forecasts is made independently of the others.
2.  **Speed:** Training one model is computationally more efficient than running a predictive loop 28 times for every single one of the 30,000+ time series during inference.
3.  **Suitability for LightGBM:** LightGBM, like many tree-based models, naturally handles multi-output regression, making it a perfect fit for this strategy.

___

<a name="feature-engineering"></a>
## 03. Feature Engineering: Giving the Model a Memory

A machine learning model has no inherent understanding of time. My primary task was to create features that would explicitly give the model a "memory" and context. I engineered hundreds of features, which can be grouped into several key categories:

*   **Temporal Features (The Model's Memory):**
    *   **Lags:** The most powerful features. I included sales from 28, 29, 30... up to 42 days prior to the prediction date. This allows the model to capture weekly seasonality and auto-regressive patterns.
    *   **Rolling Windows:** To smooth out noise and capture recent trends, I calculated rolling means and standard deviations of sales over various windows (e.g., 7, 14, 30, 60 days) on top of the lag features.

*   **Price Features (The Model's Economic Sense):**
    *   **Price Momentum:** Features that captured the change in price over time (e.g., current price vs. price one month ago).
    *   **Price Uniqueness:** A feature counting the number of unique prices an item has had, signaling promotional activity.

*   **Calendar Features (The Model's Worldly Knowledge):**
    *   **Events & Holidays:** I converted the event data (like Super Bowl or Mother's Day) into binary flags.
    *   **SNAP Days:** I included flags for days when SNAP (food assistance) benefits could be used, as EDA showed this was a huge driver of food sales.

*   **Release Date Feature:** I calculated a "days since release" feature for every product. This was critical for teaching the model to differentiate between a day of zero sales (low demand) and a day before the product was even available on shelves.

___

<a name="modeling"></a>
## 04. Modeling: A Multi-Stage Optimization Strategy

I developed the final model through a disciplined, three-stage process, starting simple and methodically adding complexity.

<a name="modeling-champion1"></a>
### Stage 1: The Robust LightGBM Baseline

The first step was to build a simple but effective baseline model. This model used only the most fundamental features:
*   Basic date features (day of week, month, year).
*   A few essential lag features (e.g., lag 28, 35, 42).
This baseline established the core data pipeline and provided an initial WRMSSE score to beat.

<a name="modeling-champion2"></a>
### Stage 2: Advanced Optimization with "Dark Magic"

This stage involved integrating the hundreds of advanced features described in the previous section. This is where the bulk of the performance gain came from, often referred to as the "dark magic" of feature engineering. I systematically added the rolling window statistics, price momentum features, and other complex variables, validating the impact of each set on my local validation score.

<a name="modeling-champion3"></a>
### Stage 3: Automated Hyperparameter Tuning

With the feature set finalized, the last step was to tune the LightGBM model's hyperparameters to squeeze out the last bit of performance. I used the **Optuna** library to run a **Bayesian Optimization** search. This automated process efficiently explored the hyperparameter space and found the optimal combination of `learning_rate`, `num_leaves`, `feature_fraction`, and regularization parameters, ensuring the model was as accurate and well-regularized as possible.

___

<a name="prediction"></a>
## 05. Evaluation & Results: Measuring Success

The competition was judged on the **Weighted Root Mean Squared Scaled Error (WRMSSE)**, a complex but fair metric perfectly suited for this business problem.

*   **RMSE:** It starts with the standard Root Mean Squared Error for each individual product.
*   **Scaled (RMSSE):** Each product's RMSE is then scaled by the error of a naive benchmark (predicting today's sales are the same as yesterday's). This allows for fair comparison between high-volume and low-volume products.
*   **Weighted (WRMSSE):** The final score is a weighted average of all the individual RMSSE scores, with the weights determined by the product's total dollar revenue. This means accuracy on high-value items is more important.

After the full three-stage optimization process, the final LightGBM model achieved a **WRMSSE score of 0.51** on a rigorous, time-based local validation set. This score demonstrates a highly accurate model, as it is over 49% better than the naive baseline and is competitive with top-ranking public solutions.

___

<a name="conclusion"></a>
## 06. Conclusion & Key Learnings

This project successfully demonstrates that a single, well-architected gradient boosting model can achieve state-of-the-art performance on a massive and complex forecasting problem.

My key takeaways from this process are:
1.  **Feature Engineering is Paramount:** The most significant performance gains came not from complex algorithms, but from thoughtful, meticulous feature engineering that encoded domain knowledge into the data.
2.  **A Robust Validation Strategy is Non-Negotiable:** A strict, time-based validation split that mimics the real world is the only way to get a reliable estimate of a model's future performance and prevent overfitting.
3.  **Efficiency is a Feature:** For a problem of this scale, techniques for memory optimization and computational efficiency are not optional extras; they are core requirements for a successful project.

Future work could explore ensembling this model with a neural network (like an LSTM) to potentially capture different patterns and further improve the score, as well as productionalizing the entire workflow in a cloud environment like AWS.
