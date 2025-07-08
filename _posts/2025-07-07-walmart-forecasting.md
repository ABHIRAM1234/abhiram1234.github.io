---
layout: post
title: "M5 Forecasting: A Production-Grade Time-Series Pipeline on Google Cloud"
image: "/posts/m5-forecasting-title-image.png"
tags: [Machine Learning, Time Series, Forecasting, LightGBM, MLOps, Google Cloud, Vertex AI, Cloud Run, BigQuery]
---

This project documents the complete, end-to-end process of tackling the M5 Forecasting challenge, one of the most comprehensive and large-scale time-series datasets available. The objective was to build a system that accurately forecasts daily sales for thousands of products across multiple stores. This report covers every step, from initial data processing and feature engineering to model training, debugging a complex cloud deployment, and creating a fully automated, production-ready forecasting pipeline on Google Cloud Platform.

# Table of Contents
- [00. Project Overview](#overview-main)
    - [Context](#overview-context)
    - [Actions](#overview-actions)
    - [Results](#overview-results)
    - [Growth & Next Steps](#overview-growth)
- [01. Data Understanding & Pre-processing](#data-overview)
- [02. The Methodology: Time-Series Regression](#methodology)
- [03. Feature Engineering: The Foundation of Forecasting](#feature-engineering)
- [04. Modeling & Iteration: The Search for a Stable Champion](#modeling)
    - [Champion Model: The Simple, Robust Baseline](#modeling-champion)
    - [Experiment 1: The Overfitting Ensemble](#modeling-experiment1)
    - [Experiment 2: The Subtle Data Bug](#modeling-experiment2)
- [05. Deployment: The Real-World MLOps Challenge](#deployment)
    - [The Training Pipeline: Conquering the Cloud](#deployment-training)
    - [The Prediction Pipeline: From Crashing to Production](#deployment-prediction)
    - [The Final Architecture: An Automated System](#deployment-architecture)
- [06. Conclusion & Key Learnings](#conclusion)

___

# 00. Project Overview <a name="overview-main"></a>

### Context <a name="overview-context"></a>

Accurate demand forecasting is one of the most critical and highest-value challenges in the retail industry. For a large retailer like Walmart, it directly impacts inventory management, supply chain logistics, labor scheduling, and ultimately, profitability. The M5 dataset, containing five years of hierarchical sales data for thousands of products, provides a realistic and complex environment to build and test a production-grade forecasting system.

### Actions <a name="overview-actions"></a>

The project was framed as a large-scale **time-series regression** problem: given the extensive history of a product, predict its daily sales for the next 28 days.

The end-to-end workflow involved:
1.  **Efficient Pre-processing** of a 58-million-row dataset, transforming it from a wide format to a memory-optimized long format.
2.  **Iterative Feature Engineering**, creating a rich feature set of time-based lags and rolling window statistics.
3.  **Systematic Model Iteration**, starting with a simple, robust baseline and experimenting with more complex ensemble models and features, leading to critical insights about the trade-off between complexity and stability.
4.  **End-to-End Cloud Deployment**, where the entire workflow was containerized with `Docker` and deployed as a fully automated pipeline on **Google Cloud Platform**, using **Vertex AI** for training and **Cloud Run** for serverless, scheduled predictions.

### Results <a name="overview-results"></a>

The final solution is a robust, automated, and scalable forecasting pipeline.
-   The champion model is a **LightGBM Regressor** that achieved a stable and highly respectable Kaggle **Private Score of 0.76090**.
-   The training pipeline successfully scales to handle the massive dataset by running on a **high-memory Vertex AI instance (`n1-highmem-8`)**.
-   The prediction pipeline is a fully automated, event-driven service built on **Cloud Run, Eventarc, Pub/Sub, and Cloud Scheduler**, which saves its forecast output to **BigQuery** weekly.
-   A **Looker Studio** dashboard provides an interactive, user-friendly interface for business users to explore the live forecast data.

### Growth/Next Steps <a name="overview-growth"></a>

The current system provides a powerful and reliable baseline. Future work could focus on:
-   **Two-Stage Modeling:** For products with many zero-sale days, a two-stage model could be implemented: a classifier to predict the *probability* of a sale, and a regressor to predict the *quantity* if a sale occurs.
-   **Granular Ensembling:** Experiment with more complex ensembles (e.g., one model per store or department) now that the deployment architecture is stable, paying close attention to hyperparameter tuning to avoid overfitting.
-   **Feature Store Integration:** Formalize the feature generation process by creating dedicated tables in BigQuery, allowing for faster and more consistent feature lookups.

___

# 01. Data Understanding & Pre-processing <a name="data-overview"></a>

The dataset comprised three core files: `sales_train_validation.csv`, `calendar.csv`, and `prices.csv`. The primary challenge was the format of the sales data. It was in a "wide" format, with each row representing a single product and 1,913 columns representing daily sales (`d_1`, `d_2`, ...).

This format is extremely memory-intensive and unsuitable for most modeling libraries. The first and most critical step was to create a preprocessing pipeline that would:
1.  **Melt the Data:** Transform the sales data into a "long" format, where each row represents a single `(product, day)` observation. This expanded the dataset to over 58 million rows.
2.  **Merge Context:** Enrich the sales data by merging it with the `calendar` and `prices` tables.
3.  **Optimize for Memory:** A custom script was developed to process the data in manageable yearly chunks, aggressively downcast data types, and save the intermediate results as highly efficient Parquet files. This was the foundation that made handling the massive dataset possible.

___

# 02. The Methodology: Time-Series Regression <a name="methodology"></a>

The task of predicting a continuous value (daily sales) makes this a classic **regression** problem. Because the data's primary dimension is time, we used techniques specifically designed for time-series forecasting.

#### Model Choice: LightGBM
A `LightGBM Regressor` was chosen for three key reasons:
1.  **Performance:** It is consistently state-of-the-art for structured, tabular data.
2.  **Efficiency:** It is significantly faster and more memory-efficient than other gradient boosting libraries, which was essential for this dataset.
3.  **Scalability:** It scales well to millions of rows and can leverage multiple CPU cores.

#### Evaluation Metric: Root Mean Squared Error (RMSE)
While the official M5 competition uses a complex metric (WRMSSE), **RMSE** was used during development and validation. It is a standard regression metric that is easy to interpret and heavily penalizes large errors, making it a good proxy for forecast accuracy. The final evaluation was based on the score produced by the Kaggle platform.

___

# 03. Feature Engineering: The Foundation of Forecasting <a name="feature-engineering"></a>

The core hypothesis of any time-series model is that the past can predict the future. The features were designed to capture historical patterns:
-   **Lag Features:** What were the sales for this exact product 28, 35, 42, 49, and 56 days ago? This captures weekly seasonality and recent trends.
-   **Rolling Window Features:** What were the average sales and standard deviation for this product over the last 7, 14, and 28 days? This smooths out noise and provides a more stable view of recent performance.

___

# 04. Modeling & Iteration: The Search for a Stable Champion <a name="modeling"></a>

The project involved a systematic, iterative approach to find the best model, which led to crucial real-world insights.

### Champion Model: The Simple, Robust Baseline <a name="modeling-champion"></a>
-   **The Action:** The first model trained was a single LightGBM model on the entire dataset using the foundational lag and rolling window features.
-   **The Result:** This model achieved a Kaggle Private Score of **0.76090**. Crucially, its public and private scores were nearly identical, proving it was extremely **stable and generalized well** to unseen data. This became our production champion.

### Experiment 1: The Overfitting Ensemble <a name="modeling-experiment1"></a>
-   **The Hypothesis:** Following the strategies of the M5 winners, training an ensemble of specialized models (one per state) should yield better results.
-   **The Result:** This model failed catastrophically, scoring **5.39** on the private leaderboard. The investigation revealed that the powerful model parameters were too complex for the smaller, state-level data slices, causing severe **overfitting**. This was a critical lesson in the trade-off between model complexity and data volume.

### Experiment 2: The Subtle Data Bug <a name="modeling-experiment2"></a>
-   **The Hypothesis:** Adding advanced price features (e.g., price momentum) should improve the model.
-   **The Result:** This model also failed with the exact same private score of **5.39**. The identical failure score pointed to a deterministic bug, not a modeling problem. A deep dive into the prediction code revealed that the recursive forecasting loop failed to handle the sparse `prices.csv` file correctly, leading to `NaN` values that cascaded and destroyed the predictions after the first few days.

___

# 05. Deployment: The Real-World MLOps Challenge <a name="deployment"></a>

Deploying this system to the cloud was a multi-stage process filled with real-world challenges that are core to the MLOps discipline.

### The Training Pipeline: Conquering the Cloud <a name="deployment-training"></a>
The first challenge was simply getting the model to train on the massive dataset in a cloud environment. The journey involved:
1.  **Containerization:** Packaging the training code and all its dependencies into a `Dockerfile`.
2.  **Debugging Dependencies:** The initial deployment failed due to an `ImportError`. The fix was to switch from a generic Python base image to a Google-provided Deep Learning base image for better stability.
3.  **Solving Configuration Errors:** The job then failed due to a `KeyError`, which was traced to a missing environment variable in the job configuration. This was solved by moving to a `config.yaml` file for explicit, repeatable deployments.
4.  **Overcoming Memory Limits:** The job then failed with an **Out of Memory** error. The solution was to upgrade the Vertex AI training instance from an `n1-standard-8` (30GB RAM) to an `n1-highmem-8` (52GB RAM), which successfully handled the memory pressure.

### The Prediction Pipeline: From Crashing to Production <a name="deployment-prediction"></a>
The prediction service had its own unique set of challenges:
1.  **The Startup Timeout Crash:** The initial deployment to Cloud Run failed because all the heavy data loading was happening at startup, causing the container to miss its health check and be terminated. The architecture was corrected by moving all heavy logic *inside* the main function, ensuring the container started instantly.
2.  **The Recursive Memory Leak:** Even after starting, the service would crash after forecasting ~20 days. This was traced to a memory leak in the recursive loop, where the history dataframe grew with each iteration. The final fix was to implement memory-efficient data loading (`usecols`) and to trim the history dataframe inside the loop, keeping the memory footprint flat and stable.
3.  **The IAM "Guest List" Problem:** The final hurdle was a `403 FORBIDDEN` error. This was solved by creating a dedicated service account for the Eventarc trigger and explicitly granting it the "Cloud Run Invoker" role, a best practice for secure event-driven architectures.

### The Final Architecture: An Automated System <a name="deployment-architecture"></a>
The final, successful deployment is a fully automated, production-grade system:
-   **Source of Truth:** **Google Cloud Storage** hosts the raw data and trained model artifact.
-   **Model Training:** **Vertex AI Custom Training** runs a containerized training job on a high-memory machine.
-   **Prediction Service:** A containerized Python application runs on **Cloud Run**, Google's serverless platform.
-   **Automation:** **Cloud Scheduler** fires a weekly event, which is sent via **Pub/Sub** and routed by **Eventarc** to trigger the Cloud Run service.
-   **Results Storage:** The final forecasts are written to a structured table in **BigQuery**.
-   **Visualization:** **Looker Studio** provides an interactive dashboard connected directly to the BigQuery results table.

___

# 06. Conclusion & Key Learnings <a name="conclusion"></a>

This project successfully transitioned from a Kaggle-style model development process to a fully deployed, production-ready MLOps pipeline. The journey provided several critical, real-world learnings:
-   **The Simplest Model is Often the Best Production Model:** Our most stable, reliable, and highest-scoring model was the simplest one. Complexity adds risk, and stability is paramount in production.
-   **Deployment is a Process of Rigorous Debugging:** Every error encountered—from dependency issues and memory limits to IAM policies and startup timeouts—is a common and expected part of a real-world cloud deployment.
-   **Architecture Matters More Than Code:** The final breakthroughs came not from changing the model, but from re-architecting the application (e.g., moving logic inside the function, loading data efficiently) to fit the constraints and best practices of a serverless cloud environment.

The result is a testament to the power of a well-architected cloud pipeline to deliver robust and scalable machine learning solutions.