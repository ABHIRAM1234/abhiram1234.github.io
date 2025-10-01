---
layout: post
title: "OptiPlay Analytics: Data-Driven Strategy for Fantasy Premier League"
image: "/posts/classification-title-img.png"
tags: [Analytics, Sports Analytics, R, Regression, Clustering, Hypothesis Testing, Feature Engineering]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Research Goals](#goals)
- [02. Data & Features](#data)
- [03. Methods](#methods)
- [04. Key Insights](#insights)
- [05. Limitations & Future Work](#limitations)
- [06. Technical Stack](#stack)
- [07. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

OptiPlay-Analytics explores how quantitative analysis can improve Fantasy Premier League (FPL) squad selection. The study evaluates player performance drivers, the effect of home-field advantage, and clustering-based player archetypes to inform transfer and captaincy decisions.

---

## <a name="goals"></a>01. Research Goals

- Identify features most predictive of FPL success (points, consistency)
- Quantify impact of home vs away performance
- Build interpretable models to support selection decisions

---

## <a name="data"></a>02. Data & Features

- Historical FPL player statistics (over 30 features)
- Player-specific metrics, match-level context, and derived indicators
- Feature groups: player performance, game context, miscellaneous

---

## <a name="methods"></a>03. Methods

- Regression: Linear and Lasso Regression for feature selection and effect sizing
- Classification: Logistic Regression for categorical outcomes
- Hypothesis testing: Chi-Square tests for association insights
- Unsupervised learning: Clustering to discover player archetypes

Reference analysis script: `Regression Analysis.R`

---

## <a name="insights"></a>04. Key Insights

- Data-driven strategies materially improve FPL decision-making
- Home-field advantage affects some teams and players more than others
- Key metrics and fitness indicators guide recruitment and captaincy

---

## <a name="limitations"></a>05. Limitations & Future Work

- Scope limitations: psychology, team chemistry, fatigue not directly modeled
- Future: expand seasons/leagues, integrate ML pipelines for near real-time insights

---

## <a name="stack"></a>06. Technical Stack

- Language: R
- Techniques: Regression (Linear/Lasso), Logistic Regression, Chi-Square, Clustering
- Workflow: Reproducible analysis in R scripts with tidy modeling approach

---

## <a name="links"></a>07. Project Links

- **GitHub Repository**: [OptiPlay-Analytics](https://github.com/ABHIRAM1234/OptiPlay-Analytics)

---

## 🚀 Why This Project Matters

Showcases sports analytics and statistical rigor for strategy optimization under uncertainty—transferable to broader applied analytics roles.


