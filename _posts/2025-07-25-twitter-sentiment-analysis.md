---
layout: post
title: "End-to-End Twitter Sentiment Analysis with PySpark, Kafka, and Spark Streaming"
image: "/posts/twitter-sentiment-analysis-architecture.png" # Recommendation: Use the architecture diagram as the main image
tags: [PySpark, Machine Learning, Data Engineering, Kafka, Spark Streaming, MLOps, NLP, Delta Lake]
---

# Table of Contents
- [00. Project Overview](#overview-main)
- [01. The Challenge: Analyzing Social Media at Scale](#challenge-main)
- [02. My Solution: A Two-Part Approach](#solution-main)
- [03. Part 1: Batch Model Training & Selection](#part1-main)
- [04. Part 2: Real-Time Inference Pipeline](#part2-main)
- [05. Key Technologies Used](#tech-main)
- [06. Project Links](#links-main)
- [07. Key Takeaways & Learnings](#takeaways-main)

___

<a name="overview-main"></a>
## 00. Project Overview

This project demonstrates a complete, end-to-end system for sentiment analysis on Twitter data. The solution is architected to handle both large-scale batch processing for model training and real-time stream processing for live inference.

At its core, the project uses **PySpark** for distributed model training on the **1.6 million tweet Sentiment140 dataset**. The best-performing model is then deployed into a robust, containerized data pipeline orchestrated with **Apache Kafka**, **Spark Structured Streaming**, and **Docker**, with final results stored in a **Delta Lake** for reliable analytics.

#### System Architecture
<p align="center">
  <img src="https://raw.githubusercontent.com/ABHIRAM1234/pyspark-etl-twitter/main/images/flow.png" width="800" alt="Real-Time Sentiment Analysis Architecture Diagram">
</p>

___

<a name="challenge-main"></a>
## 01. The Challenge: Analyzing Social Media at Scale

Businesses and organizations need to understand public sentiment in real-time to monitor brand perception, track marketing campaign success, and respond to customer feedback. However, the sheer volume and velocity of data from platforms like Twitter present a significant engineering challenge. A viable solution requires a system that can not only build an accurate predictive model but also deploy it in a way that is scalable, fault-tolerant, and capable of handling a continuous stream of data.

___

<a name="solution-main"></a>
## 02. My Solution: A Two-Part Approach

To tackle this challenge, I broke the project into two distinct but connected phases:

1.  **Batch Model Training:** First, I needed to find the most accurate and efficient model for the task. This involved a systematic process of feature engineering, training, and evaluating multiple machine learning classifiers in a distributed environment using PySpark MLlib.
2.  **Real-Time Inference Pipeline:** With a trained model in hand, the next step was to build a production-grade streaming pipeline to serve the model for live predictions. This focused on data engineering principles to ensure the system was robust and scalable.

___

<a name="part1-main"></a>
## 03. Part 1: Batch Model Training & Selection

The goal of this phase was to rigorously determine the best model for sentiment classification.

### Methodology
I used PySpark to preprocess the 1.6 million tweet dataset and then evaluated several models and feature sets to identify the optimal combination.

*   **Models Evaluated:** Logistic Regression, Support Vector Machines, Naive Bayes, Random Forest, Decision Tree.
*   **Feature Engineering:** I tested multiple techniques, including Hashing TF-IDF vs. CountVectorizer + TF-IDF, the inclusion of N-grams (1, 2, and 3-grams), and the use of ChiSqSelector for automated feature selection.

### Key Results
The analysis concluded that a **Logistic Regression** classifier, when combined with **CountVectorizer, TF-IDF, and 1-2-3-grams**, yielded the highest performance, achieving an **F1-Score of 0.808**. This demonstrated that a simpler, more interpretable model could outperform more complex ones with the right feature engineering.

<p align="center">
  <b>Table 1: Model Accuracy Across Different Feature Sets</b><br>
  <img src="https://raw.githubusercontent.com/ABHIRAM1234/twitter-sentiment-analysis-pyspark/main/images/features.png" width="700"/>
</p>
<br>
<p align="center">
  <b>Table 2: Detailed Performance Metrics for Final Models</b><br>
  <img src="https://raw.githubusercontent.com/ABHIRAM1234/twitter-sentiment-analysis-pyspark/main/images/summary.png" width="700"/>
</p>

This training process was designed to run at scale on a cloud platform like **Google Cloud Dataproc**.

___

<a name="part2-main"></a>
## 04. Part 2: Real-Time Inference Pipeline

This phase focused on deploying the trained Logistic Regression model into a live data pipeline.

### Architecture Deep Dive
*   **Containerization (`Docker`):** The entire data ingestion component, consisting of the Twitter API producer and the Kafka broker, is containerized with Docker. This ensures portability and simplifies deployment.
*   **Message Brokering (`Apache Kafka`):** Live tweets are pushed into a Kafka topic. Kafka acts as a durable, scalable buffer between the data source (Twitter) and the stream processor (Spark), decoupling the system and preventing data loss.
*   **Stream Processing (`Spark Structured Streaming`):** A Spark application continuously consumes messages from the Kafka topic. For each incoming tweet, it applies the pre-trained PySpark MLlib model to classify the sentiment in real-time.
*   **Reliable Storage (`Delta Lake`):** The final results (tweet text, timestamp, and predicted sentiment) are written to a Delta Lake. I chose Delta Lake over a standard data lake because it provides ACID transactions, schema enforcement, and time travel capabilities, ensuring data integrity and reliability for downstream analytical queries.

___

<a name="tech-main"></a>
## 05. Key Technologies Used

*   **Data Processing & ML:** Python, PySpark, Pandas, Scikit-learn, Spark MLlib, NLTK
*   **Data Engineering:** Apache Kafka, Spark Structured Streaming, Delta Lake, Docker
*   **Databases:** MongoDB, PostgreSQL
*   **Cloud & MLOps:** Google Cloud Dataproc, Git, GitHub Actions (for CI/CD)

___

<a name="links-main"></a>
## 06. Project Links

This project is divided into two repositories, one for the model training and another for the real-time ETL pipeline.

1.  **[GitHub Repo: Model Training & Evaluation](https://github.com/ABHIRAM1234/twitter-sentiment-analysis-pyspark)**
2.  **[GitHub Repo: Real-Time ETL Pipeline](https://github.com/ABHIRAM1234/pyspark-etl-twitter)**

___

<a name="takeaways-main"></a>
## 07. Key Takeaways & Learnings

*   **Feature Engineering is King:** A well-tuned feature set allowed a simpler model like Logistic Regression to outperform more complex ones, reinforcing the importance of feature engineering in the ML lifecycle.
*   **The Power of Decoupling:** Using Kafka as a message bus is critical for building robust streaming systems. It isolates the data producer from the consumer, allowing each to be scaled, updated, or fail independently without bringing down the entire pipeline.
*   **Reliability in the Data Lake:** Implementing Delta Lake was a key learning. It solves many of the common reliability and data quality issues of traditional data lakes, making it possible to build production-grade analytics on streaming data.
*   **End-to-End Thinking:** This project was a valuable exercise in connecting the two worlds of machine learning and data engineering. Building a great model is only half the battle; creating a system to serve it reliably at scale is what delivers true business value.