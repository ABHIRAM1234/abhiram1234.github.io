---
layout: post
title: "DataHarbor Internship Trends: End-to-End AWS Data Pipeline to Tableau"
image: "/posts/dataharbor-title-img.png"
tags: [Data Engineering, AWS, ETL, Web Scraping, Python, Glue, Lambda, EventBridge, S3, RDS, Tableau]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Problem & Objectives](#problem-objectives)
- [02. Architecture & Flow](#architecture)
- [03. Data Ingestion & Scraping](#ingestion)
- [04. Transformation with AWS Glue](#transformation)
- [05. Storage & Load (S3 → RDS)](#storage-load)
- [06. Dashboard & Insights](#dashboard)
- [07. Technical Stack](#tech-stack)
- [08. What I Did / Key Contributions](#contributions)
- [09. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

Designed and implemented an automated data pipeline to collect internship/job listings, transform them using serverless data engineering on AWS, and publish curated tables for analytics in Tableau. The pipeline is resilient, event-driven, and modular, enabling rapid iteration on scraping logic and transformations without manual orchestration.

---

## <a name="problem-objectives"></a>01. Problem & Objectives

- Consolidate internship postings across sources into a structured, queryable dataset
- Automate the end-to-end ELT/ETL flow with minimal manual steps
- Enable trend analysis by category, region, and company in an interactive dashboard

---

## <a name="architecture"></a>02. Architecture & Flow

High-level event-driven pipeline:

```
Custom Python Scraper → S3 (raw CSV)
        ↓ (S3 event)
AWS Lambda (trigger crawler) → AWS Glue Crawler → Glue Data Catalog
        ↓ (Crawler completion event)
AWS EventBridge → AWS Lambda (trigger ETL) → AWS Glue Job (PySpark)
        ↓
AWS RDS (MS SQL Server) ← Curated data in S3
        ↓
Tableau Dashboard (live connection / extracts)
```

Key traits: serverless triggers, catalog-driven schema discovery, centralized metadata, and scalable PySpark transforms.

---

## <a name="ingestion"></a>03. Data Ingestion & Scraping

- Python-based web scraper deposits raw CSVs to S3
- New object create events in S3 invoke a Lambda to run the Glue Crawler
- The crawler registers schemas in the Glue Data Catalog for downstream jobs

Scraper reference: `WEB_SCRAPPER.py` in the repo.

---

## <a name="transformation"></a>04. Transformation with AWS Glue

- PySpark ETL cleans, normalizes, and deduplicates postings
- Standardizes fields (title, company, location, category, posted date)
- Applies business rules to tag internships and derive categories
- Outputs curated tables back to S3 and loads into RDS

---

## <a name="storage-load"></a>05. Storage & Load (S3 → RDS)

- Curated datasets are written to S3 in CSV/Parquet and ingested into AWS RDS (MS SQL Server)
- RDS acts as the analytics serving layer for BI tools

---

## <a name="dashboard"></a>06. Dashboard & Insights

Tableau dashboard explores:
- Job distribution by category and region
- Top companies by internship openings
- Time trends and seasonality in postings

Sample packaged workbook is included in the repository (`Sample_Dashboard.twbx`).

---

## <a name="tech-stack"></a>07. Technical Stack

- AWS: S3, Glue Crawler, Glue Jobs (PySpark), Lambda, EventBridge, RDS (MS SQL Server)
- Data: CSV/Parquet in S3 curated zones
- Languages: Python, PySpark
- BI: Tableau

---

## <a name="contributions"></a>08. What I Did / Key Contributions

- Built event-driven orchestration using S3 → Lambda → Glue → EventBridge
- Authored PySpark transformations for data standardization and deduplication
- Set up Glue Crawler + Data Catalog for schema management
- Automated load to RDS to power Tableau analytics

---

## <a name="links"></a>09. Project Links

- **GitHub Repository**: [DataHarbor-Internship-Trends](https://github.com/ABHIRAM1234/DataHarbor-Internship-Trends)
- **Packaged Tableau Workbook**: Included in repo (`Sample_Dashboard.twbx`)

---

## 🚀 Why This Project Matters

Demonstrates production-grade, serverless data engineering on AWS with a clear analytics outcome. The design emphasizes maintainability, observability, and scalability—skills directly applicable to real-world data platforms.


