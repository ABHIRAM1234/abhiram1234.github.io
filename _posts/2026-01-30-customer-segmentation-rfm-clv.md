---
layout: post
title: "Customer Segmentation RFM & CLV: Online Retail Analytics"
image: "/posts/customer-segmentation-img.png"
tags: [Analytics, Retail, RFM, Customer Lifetime Value, Clustering, Python, K-means, Segmentation]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Why This Problem Matters](#why-this-matters)
- [02. Key Concepts: RFM and CLV Explained](#concepts)
- [03. My Step-by-Step Thought Process](#thought-process)
- [04. Data & Features](#data)
- [05. Methods in Detail](#methods)
- [06. Key Insights & How to Act on Them](#insights)
- [07. Limitations & Future Work](#limitations)
- [08. Technical Stack](#stack)
- [09. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

**What this project is (in plain English):**  
I analyzed **online retail transaction data**—who bought what, when, and how much—to answer two business questions: (1) **Who are my best customers, who is at risk, and who has gone quiet?** and (2) **How much revenue can I expect from each customer in the future?** To do that, I used two classic analytics frameworks: **RFM (Recency, Frequency, Monetary)** for segmentation and **Customer Lifetime Value (CLV)** for valuing customers and guiding marketing spend.

**At a glance**
- **Objective:** Segment customers by behavior (RFM) and estimate their future value (CLV) so marketing and retention efforts can be targeted and efficient.
- **Approach:** Compute RFM from transaction history → cluster customers into segments (e.g. Champions, At Risk, Dormant) → build descriptive, probabilistic, and predictive CLV models.
- **Output:** Actionable segments and CLV estimates to prioritize retention, win-back, and acquisition.
- **Repo:** [Customer-Segmentation-RFM-CLV](https://github.com/ABHIRAM1234/Customer-Segmentation-RFM-CLV); reference notebook: `online-retail-data-clustering.ipynb`.

---

## <a name="why-this-matters"></a>01. Why This Problem Matters

**Business context:**  
Retailers and e-commerce businesses cannot treat all customers the same. Some customers buy often and spend a lot; others bought once and never returned. If you send the same generic campaign to everyone, you waste budget on low-value or dormant customers and under-invest in high-value or at-risk ones.  

**What we need:**  
(1) A **simple, interpretable way to group customers** by how they behave (recent? frequent? big spenders?).  
(2) A **sense of how much each customer (or segment) is worth over time** so we can decide where to put retention offers, win-back campaigns, and acquisition spend.  

**Why RFM and CLV:**  
RFM is a proven, easy-to-explain framework that uses only transaction history (no demographics required) to create segments. CLV turns “how much have they spent?” into “how much are they likely to spend in the future?” so we can prioritize actions and measure ROI.

---

## <a name="concepts"></a>02. Key Concepts: RFM and CLV Explained

**RFM (Recency, Frequency, Monetary)**  
- **Recency:** How long since the customer’s last purchase? (e.g. days or weeks.) Lower recency = more recent = generally better.  
- **Frequency:** How many transactions (orders) did they make in the analysis window? Higher = more engaged.  
- **Monetary:** Total amount spent (or average order value) in the window. Higher = more valuable.  

We typically **score** each dimension (e.g. 1–5 or quintiles) and combine scores to label segments: e.g. “Champions” (high on all three), “At Risk” (were good, but recency has dropped), “Dormant” (low recency and frequency).

**Customer Lifetime Value (CLV)**  
CLV is the expected value (often revenue or profit) we get from a customer over their relationship with us.  
- **Descriptive CLV:** Uses past behavior only (e.g. average revenue per customer so far).  
- **Probabilistic CLV:** Uses probability models (e.g. when will they buy again? will they churn?) to project behavior.  
- **Predictive CLV:** Uses ML or statistical models to forecast future revenue or profit per customer (or segment).

In this project I used the online retail dataset to illustrate all three: simple descriptive metrics, probabilistic ideas (e.g. repeat purchase, retention), and predictive modeling where applicable.

---

## <a name="thought-process"></a>03. My Step-by-Step Thought Process

I approached the project as a clear pipeline: understand the data → build behavior metrics (RFM) → segment → value customers (CLV) → derive actions.

---

### Step 1: Understand the Data and the Business Question

**What I did:**  
I loaded the online retail transaction data and inspected its structure: customer IDs, order dates, product codes, quantities, prices, and any other key fields.

**Why:**  
Without knowing what we have (e.g. do we have one row per order, or per line item? Is there a customer identifier? Are there returns or cancellations?), we cannot define RFM or CLV correctly.

**Decisions I made:**  
- Define the **analysis window** (e.g. last 12 months or all available history) so Recency and Frequency are comparable across customers.  
- Decide the **unit of “transaction”** (e.g. one row per order vs. per line) and aggregate to one row per order if needed, then roll up to customer level.  
- Handle **data quality:** missing customer IDs, invalid dates, negative quantities or prices, and (if relevant) returns. I documented how I treated these so results are reproducible.

---

### Step 2: Compute RFM at the Customer Level

**What I did:**  
For each customer, I computed:  
- **Recency:** Days (or weeks) since their most recent order.  
- **Frequency:** Number of orders in the window.  
- **Monetary:** Sum of (quantity × price) or total revenue in the window.

**Why:**  
These three numbers summarize “how recently they bought,” “how often they buy,” and “how much they spend.” They are the inputs for both segmentation and (later) CLV.

**Decisions I made:**  
- Use a **fixed snapshot date** (e.g. max date in the data) so Recency is comparable.  
- Choose whether Monetary is **total revenue** or **average order value**; I used total revenue for CLV-oriented segments.  
- **Score RFM** (e.g. quintiles 1–5): higher score = better for Recency (more recent), Frequency, and Monetary. This makes it easy to name segments (e.g. 5-5-5 = best customers).

---

### Step 3: Segment Customers Using RFM (and Optional Clustering)

**What I did:**  
I created segments in two complementary ways:  
- **Rule-based:** Using RFM scores (e.g. Champions = high on all; At Risk = high F and M but low R; Dormant = low R and F).  
- **Clustering:** Used K-means (or similar) on standardized RFM (or R, F, M directly) to find data-driven groups, then labeled them by their typical RFM profile.

**Why:**  
Rule-based segments are easy to explain and act on; clustering can surface segments we didn’t name by hand (e.g. “recent one-time big spenders”). I used both to get a complete picture.

**Decisions I made:**  
- **Standardize** R, F, M before clustering (e.g. z-scores) so scale doesn’t dominate.  
- Choose **number of clusters** (elbow method, silhouette, or business-driven K).  
- Map cluster centroids back to RFM to name segments (e.g. “high recency, low frequency” = new or occasional buyers).

---

### Step 4: Build Customer Lifetime Value (Descriptive → Predictive)

**What I did:**  
- **Descriptive:** For each segment (or customer), computed historical metrics: total revenue, average order value, average orders per year. This is “value so far” and simple benchmarks.  
- **Probabilistic:** Considered repeat-purchase and retention (e.g. probability of next purchase, expected time to next order). The notebook uses or references these ideas where the data supports it.  
- **Predictive:** Where applicable, used regression or other models to predict future revenue (e.g. next 12 months) per customer or segment using RFM and other features.

**Why:**  
Descriptive CLV tells us who has been valuable; probabilistic and predictive CLV tell us who *will* be valuable so we can prioritize retention and win-back.

**Decisions I made:**  
- Define **CLV horizon** (e.g. next 12 months or 3 years) and whether we model revenue or profit.  
- Use **segment-level CLV** when customer-level data is sparse; use **customer-level** when we have enough history and want personalized actions.

---

### Step 5: Turn Segments and CLV Into Actions

**What I did:**  
I summarized which segments exist, their size, their RFM profile, and their (descriptive or predictive) CLV. Then I outlined **what to do** with each:  
- **Champions (high R, F, M):** Reward, loyalty program, upsell; protect from churn.  
- **At Risk (low R, high F, M):** Win-back campaign, personalized offer.  
- **Dormant (low R, low F):** Re-activation campaign or lower priority.  
- **New / Potential (high R, low F):** Nurture with second-purchase incentives.

**Why:**  
The goal of the project is not only to describe segments but to **inform marketing and resource allocation**. CLV helps decide how much to spend on each segment.

---

## <a name="data"></a>04. Data & Features

- **Data source:** Online retail transaction dataset (e.g. UCI or similar): transaction-level records with customer ID, date, product, quantity, unit price.  
- **Subject area:** Business, Retail.  
- **Feature types:** Integer (quantity, counts), Real (prices, amounts, days).  
- **Derived:** Customer-level Recency (days), Frequency (order count), Monetary (total revenue); optional: average order value, average days between orders.  
- **Data quality:** I handled missing or invalid customer IDs, dropped or adjusted negative quantities/prices, and used a consistent analysis window so RFM is comparable across customers.

---

## <a name="methods"></a>05. Methods in Detail

- **RFM computation:** Per customer, Recency = days since last order; Frequency = number of orders; Monetary = sum(quantity × price) in the analysis window. Optionally scored into quintiles (1–5).  
- **Segmentation:**  
  - Rule-based: Segment definitions from RFM score combinations (e.g. 5-5-5 = Champions).  
  - Clustering: K-means (Scikit-learn) on standardized R, F, M; elbow/silhouette to choose K; labels mapped to interpretable segment names.  
- **CLV:**  
  - Descriptive: Mean revenue per customer (or per segment), AOV, order rate.  
  - Probabilistic: Notion of repeat purchase and retention (as in the notebook).  
  - Predictive: Regression (or other) model predicting future revenue from RFM and other covariates.  
- **Visualization:** Distribution of R, F, M; segment sizes; segment RFM heatmaps; CLV by segment. All in the Jupyter notebook for reproducibility.

---

## <a name="insights"></a>06. Key Insights & How to Act on Them

- **Segment clarity:** RFM + clustering surfaces distinct groups: high-value loyal (Champions), slipping (At Risk), and dormant (Dormant). Each gets a clear action: retain, win back, or re-activate with lower priority.  
- **CLV guides spend:** Segments with high descriptive or predictive CLV justify more retention and loyalty investment; low-CLV or dormant segments get cheaper, broad win-back or are deprioritized.  
- **Recency is critical:** Customers who haven’t bought recently (high Recency = long time since purchase) are at risk of churn; pairing Recency with Frequency and Monetary tells us *who* is worth saving first.

---

## <a name="limitations"></a>07. Limitations & Future Work

- **Data and window:** Results depend on the analysis window and data quality; long gaps or incomplete history can bias RFM and CLV.  
- **External factors:** Seasonality, campaigns, and economic shocks are not modeled explicitly; incorporating them would improve predictive CLV.  
- **Next steps:** Extend to more product lines or channels; integrate probabilistic CLV libraries (e.g. BG/NBD, Gamma-Gamma); automate segment refresh and reporting (e.g. monthly RFM + CLV dashboards).

---

## <a name="stack"></a>08. Technical Stack

- **Language:** Python.  
- **Libraries:** Pandas (data and RFM), NumPy, Matplotlib/Seaborn (plots), Scikit-learn (K-means, standardization, regression), SciPy if needed.  
- **Workflow:** Single Jupyter notebook (`online-retail-data-clustering.ipynb`) with clear sections: load → clean → RFM → segment → CLV → insights.

---

## <a name="links"></a>09. Project Links

- **GitHub Repository:** [Customer-Segmentation-RFM-CLV](https://github.com/ABHIRAM1234/Customer-Segmentation-RFM-CLV)

---

## Why This Project Matters (For Recruiters)

This project shows **end-to-end retail analytics**: from raw transactions to interpretable segments (RFM) and to value-based prioritization (CLV). It demonstrates that I can frame a business problem, choose appropriate methods (RFM, clustering, descriptive and predictive CLV), implement them in code, and translate outputs into clear actions—skills that apply directly to marketing analytics, CRM, and growth roles.
