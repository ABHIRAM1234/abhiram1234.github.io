---
layout: post
title: "Network Security ML System: MLOps for Malicious URL Detection"
image: "/posts/classification-title-img.png"
tags: [MLOps, Network Security, Machine Learning, Python, FastAPI, Streamlit, Airflow, MLflow, Docker, AWS, CI/CD]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Why This Problem Matters](#why-this-matters)
- [02. Key Concepts: Malicious URL Detection & MLOps](#concepts)
- [03. My Step-by-Step Thought Process](#thought-process)
- [04. Data & Features](#data)
- [05. Methods & Pipeline](#methods)
- [06. Key Results & Components](#results)
- [07. Limitations & Future Work](#limitations)
- [08. Technical Stack](#stack)
- [09. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

**What this project is (in plain English):**  
I built an **end-to-end MLOps system** that **detects and classifies malicious URLs**—identifying harmful links (phishing, malware, social engineering) so users and systems can block or flag them. The system classifies each URL as **Malicious**, **Suspicious**, or **Safe** using a **machine learning model** (e.g. Random Forest) trained on **30 engineered URL features** (IP address in URL, length, shortening services, SSL state, etc.). The solution is **production-style**: **data ingestion** from **MongoDB**, **training and retraining** orchestrated by **Apache Airflow**, **experiment tracking** with **MLflow** on **AWS**, **real-time single-URL predictions** via a **Streamlit** app, and **batch predictions** via a **FastAPI** API (`/train` and `/predict`). The whole app is **containerized with Docker**, deployed on **AWS EC2**, and wired to a **GitHub Actions** CI/CD pipeline (build → push to **AWS ECR** → deploy to EC2). So someone who doesn’t know the project can understand: we take URLs → extract features → classify → serve predictions in real time and in batch, with full MLOps (orchestration, tracking, CI/CD).

**At a glance**
- **Objective:** Detect and classify URLs as Malicious, Suspicious, or Safe to improve web security.
- **Approach:** 30 URL features → ML classification (Random Forest, GridSearchCV) → Streamlit (single URL) + FastAPI (batch + /train) → Airflow DAGs for training/prediction → MLflow on AWS → Docker + EC2 + GitHub Actions CI/CD.
- **Code:** [GitHub Repository](https://github.com/ABHIRAM1234/Network-Security-ML-System) (forked from [Neeraj876/Network-Security-ML-System](https://github.com/Neeraj876/Network-Security-ML-System)).
- **Live demo:** [Streamlit app](https://network-security-system-mlops-6nshvkcv8cxrngosbdlrhb.streamlit.app/) for real-time single-URL predictions.

---

## <a name="why-this-matters"></a>01. Why This Problem Matters

**Business context:**  
Malicious URLs are used in **phishing**, **social engineering**, and **malware** campaigns, causing financial loss, data breaches, and reputational damage. Manual review of links doesn’t scale. We need an **automated, scalable** system that (1) **classifies** URLs in real time (e.g. in a browser or email client) and (2) **processes large batches** (e.g. log analysis, threat intelligence). The system must be **reliable** (retraining, monitoring) and **deployable** (containers, cloud, CI/CD)—i.e. full **MLOps**.

**What we need:**  
- **Real-time single-URL prediction** for user-facing tools (e.g. “Is this link safe?”).  
- **Batch prediction** for bulk analysis (e.g. CSV upload → JSON results).  
- **Automated training/retraining** so the model stays up to date as new threats appear.  
- **Experiment tracking and reproducibility** (MLflow, versioned data/model artifacts).  
- **CI/CD** so code and model updates flow from repo to production (Docker, ECR, EC2, GitHub Actions).

**Why this architecture:**  
I used **Streamlit** for a simple UI (single URL), **FastAPI** for a robust API (batch + `/train`), **Airflow** to orchestrate training and batch prediction DAGs, **MLflow** on AWS for metrics and artifacts, and **Docker + EC2 + GitHub Actions** for deployment and CI/CD. That gives both “demo” and “production” paths and shows end-to-end MLOps.

---

## <a name="concepts"></a>02. Key Concepts: Malicious URL Detection & MLOps

**Malicious URL detection**  
We do **not** use raw URL strings as input. We **engineer features** from each URL (e.g. has IP? length? shortening service? SSL state? domain age?) and train a **classifier** (e.g. Random Forest) to predict one of **Malicious**, **Suspicious**, or **Safe**. The dataset contains labeled URLs with 30 such features; we validate schema, check for drift, preprocess (e.g. KNNImputer), train, and evaluate (Precision, Recall, F1).

**MLOps (ML Operations)**  
End-to-end lifecycle: **data ingestion** (e.g. MongoDB → feature store), **training** (with validation, drift checks, preprocessing), **experiment tracking** (MLflow: metrics, artifacts, models), **serving** (Streamlit + FastAPI), **orchestration** (Airflow: training DAG, batch-prediction DAG), and **deployment** (Docker, AWS ECR, EC2, GitHub Actions). The goal is **reproducible**, **automated**, and **monitorable** ML in production.

**Two serving modes**  
- **Real-time single URL:** User enters one URL in the **Streamlit** app → model predicts Malicious/Suspicious/Safe.  
- **Batch:** User uploads a CSV of URLs (or feature rows) to **FastAPI** `/predict` → API returns predictions (e.g. JSON). **FastAPI** `/train` triggers (or kicks off) model training with current data (e.g. from MongoDB or feature store).

**CI/CD**  
**GitHub Actions** builds the Docker image, pushes to **AWS ECR**, and deploys to **AWS EC2** (e.g. self-hosted runner or target instance). Code and config changes in the repo flow to production without manual deploy steps.

---

## <a name="thought-process"></a>03. My Step-by-Step Thought Process

I approached the project as: define the security task → get data and features → build the ML pipeline → add MLOps (orchestration, tracking, serving) → deploy with CI/CD.

---

**Step 1: Define the task and the evaluation metric**  
The task is **multi-class classification**: Malicious, Suspicious, Safe. I used standard metrics (**Precision**, **Recall**, **F1-score**) and tracked them in **MLflow** so we can compare runs and models. I ensured the **evaluation pipeline** (train/test split, no leakage) is explicit and reproducible.

**Step 2: Data ingestion and feature set**  
Data is stored in **MongoDB**; the pipeline **ingests** it, validates **schema** (required columns, types), and exports to a **feature store** (or processed dataset) for training. The dataset has **30 URL-derived features** (e.g. `having_IP_Address`, `URL_Length`, `Shortening_Service`, `SSLfinal_State`). I documented each feature so someone who doesn’t know the project can understand what the model uses.

**Step 3: Data validation, drift, and preprocessing**  
I added **data validation** (schema checks, numerical correctness) and **drift detection** (statistical tests, reports) to monitor input quality over time. **Preprocessing** includes handling missing values (e.g. **KNNImputer**), saving the **preprocessor** (e.g. pickle) so the same transform is applied at inference. Train/test split is done without leakage (e.g. time-based or random with fixed seed).

**Step 4: Model training and selection**  
I trained several classifiers (Random Forest, Gradient Boosting, Decision Tree, Logistic Regression, AdaBoost) and used **GridSearchCV** for hyperparameter tuning. I compared them on Precision, Recall, and F1; selected the best (e.g. **Random Forest**); saved the final model and preprocessor; and logged everything in **MLflow** (metrics, artifacts, model) on **AWS** (e.g. S3 backend, EC2-hosted UI).

**Step 5: Serving: Streamlit + FastAPI**  
I exposed two interfaces: (1) **Streamlit** for **single-URL** real-time prediction (user-friendly), and (2) **FastAPI** for **batch** predictions (CSV upload → JSON) and for **triggering training** (`/train`). FastAPI also serves as the backend that Airflow or other systems can call for batch jobs.

**Step 6: Orchestration with Airflow**  
I defined **Airflow DAGs**: one for the **training pipeline** (ingest → validate → preprocess → train → log to MLflow) and one for **batch prediction** (e.g. pull data → predict → write results). That automates retraining and batch scoring on a schedule or on demand.

**Step 7: Deployment and CI/CD**  
I **containerized** the app with **Docker**, stored images in **AWS ECR**, and ran the app on **AWS EC2**. **GitHub Actions** builds the image on push, pushes to ECR, and deploys to EC2 (or runs tests), so the repo is the single source of truth for code and deployment.

---

## <a name="data"></a>04. Data & Features

- **Data source:** URL data with labels (Malicious / Suspicious / Safe) and **30 engineered features** per URL. Data is ingested from **MongoDB** and (optionally) exported to a feature store for training.
- **Schema & validation:** Pipeline validates that all required columns are present and numerically valid; **drift detection** (e.g. statistical tests) produces reports to monitor consistency over time.
- **Key features (examples):**  
  - `having_IP_Address` — URL contains IP instead of domain  
  - `URL_Length` — longer URLs often associated with malicious content  
  - `Shortening_Service` — e.g. bit.ly  
  - `having_At_Symbol`, `double_slash_redirecting`  
  - `Prefix_Suffix`, `having_Sub_Domain`, `SSLfinal_State`  
  - `Domain_registration_length`, `Favicon`, `port`, `HTTPS_token`  
  - `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email`  
  - `Abnormal_URL`, `Redirect`, `on_mouseover`, `RightClick`, `popUpWindow`, `Iframe`  
  - `age_of_domain`, `DNSRecord`, `web_traffic`, `Page_Rank`, `Google_Index`  
  - `Links_pointing_to_page`, `Statistical_report`  
- **Preprocessing:** Missing values handled (e.g. **KNNImputer**); preprocessor saved (e.g. pickle) for use at inference. Train/test split with no leakage.

---

## <a name="methods"></a>05. Methods & Pipeline

- **Data ingestion:** MongoDB → export to feature store / training dataset; split into train/test.
- **Validation & drift:** Schema validation; numerical checks; drift detection with reports.
- **Transformation:** Preprocessing (e.g. KNNImputer); save preprocessor; produce transformed arrays for training.
- **Modeling:** Multiple classifiers (Random Forest, Gradient Boosting, Decision Tree, Logistic Regression, AdaBoost); **GridSearchCV** for hyperparameter tuning; best model selected by Precision/Recall/F1; final model and preprocessor saved (e.g. pickle).
- **Experiment tracking:** **MLflow** on AWS (S3 for artifacts, EC2 for UI); log metrics (F1, Precision, Recall), artifacts, and model.
- **Serving:**  
  - **Streamlit:** Single URL → feature extraction (or use precomputed features) → model → Malicious/Suspicious/Safe.  
  - **FastAPI:** `/train` — trigger training (or enqueue Airflow); `/predict` — upload CSV (URLs or features) → return predictions (JSON).
- **Orchestration:** **Apache Airflow** DAGs for training pipeline and batch prediction pipeline (scheduled or manual).
- **Deployment:** **Docker** image → **AWS ECR** → **AWS EC2**; **GitHub Actions** for build, push, and deploy.

---

## <a name="results"></a>06. Key Results & Components

- **End-to-end MLOps pipeline:** Data ingestion → validation → preprocessing → training → tracking (MLflow) → serving (Streamlit + FastAPI) → orchestration (Airflow) → deployment (Docker, EC2, CI/CD).
- **Dual interface:** Real-time **Streamlit** app for single-URL checks; **FastAPI** for batch predictions and training trigger.
- **Automated retraining:** Airflow-driven training pipeline so the model can be updated on a schedule or when new data is available.
- **Reproducibility:** MLflow tracks metrics and artifacts; schema and drift checks support data quality; Docker and GitHub Actions ensure consistent builds and deploys.
- **Live demo:** [Streamlit app](https://network-security-system-mlops-6nshvkcv8cxrngosbdlrhb.streamlit.app/) for instant URL safety checks.

---

## <a name="limitations"></a>07. Limitations & Future Work

- **Feature set:** Model depends on the 30 URL features; evasive or novel attack patterns may need additional features or different data sources (e.g. page content, redirect chains).
- **Label quality:** Performance is bounded by label quality (Malicious/Suspicious/Safe); noisy or outdated labels can limit accuracy.
- **Operational scope:** MongoDB, Airflow, MLflow, and AWS setup (EC2, S3, ECR) require appropriate access and cost control; the README’s “How to Run” describes local and deployed options.
- **Future work:** Richer features (e.g. redirect chains, content-based); online learning or periodic retraining with new threat data; model monitoring (accuracy, drift) in production; optional API auth and rate limiting for FastAPI.

---

## <a name="stack"></a>08. Technical Stack

- **Frontend:** Streamlit (single-URL real-time prediction).
- **Backend:** FastAPI (/train, /predict batch); Python.
- **Modeling:** Random Forest (primary), plus Gradient Boosting, Decision Tree, Logistic Regression, AdaBoost; GridSearchCV; scikit-learn.
- **Data:** MongoDB (ingestion); schema validation; drift detection; KNNImputer; feature store / CSV for training.
- **Orchestration:** Apache Airflow (training DAG, batch-prediction DAG).
- **Experiment tracking:** MLflow (metrics, artifacts, model registry); backend on AWS (S3, EC2).
- **CI/CD:** GitHub Actions (build, test, push to ECR, deploy to EC2).
- **Containerization & cloud:** Docker, AWS ECR, AWS S3, AWS EC2.

---

## <a name="links"></a>09. Project Links

- **[GitHub Repository](https://github.com/ABHIRAM1234/Network-Security-ML-System)** — MLOps solution for malicious URL detection (Streamlit, FastAPI, Airflow, MLflow, Docker, AWS)
- **[Streamlit live demo](https://network-security-system-mlops-6nshvkcv8cxrngosbdlrhb.streamlit.app/)** — Real-time single-URL prediction
- **Original project:** [Neeraj876/Network-Security-ML-System](https://github.com/Neeraj876/Network-Security-ML-System) (forked from)

---

## Why This Project Matters (For Recruiters)

This project demonstrates **end-to-end MLOps** in a **security** context: from data ingestion (MongoDB) and feature engineering (30 URL features) to training (multiple classifiers, GridSearchCV), experiment tracking (MLflow on AWS), dual serving (Streamlit + FastAPI), orchestration (Airflow), and deployment (Docker, ECR, EC2, GitHub Actions). It shows the ability to build **production-style** ML systems with **reproducibility**, **automation**, and **clear interfaces** for both real-time and batch use cases—skills that transfer to other MLOps and security ML roles.
