---
layout: post
title: "Sarcasm Detection in News Headlines"
image: "/posts/sarcasm-detection.jpg"
tags: [NLP, Machine Learning, Logistic Regression, TF-IDF, Word2Vec, BERT, Text Classification, Python]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Dataset & Problem](#dataset-problem)
- [02. Data Cleaning Pipeline](#data-cleaning)
- [03. Feature Engineering](#feature-engineering)
- [04. Modeling & Evaluation](#modeling)
- [05. Interpreting Features](#feature-importance)
- [06. Technical Stack](#tech-stack)
- [07. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

Built a sarcasm detector for short news headlines using classic NLP and lightweight models. The goal was to understand linguistic cues of sarcasm (not just source websites) and achieve strong accuracy with interpretable features.

Source reference: [Sarcasm Detection in News Headlines](https://hi5sahil.github.io/Sarcasm-Detection-in-News-Headlines/)

---

## <a name="dataset-problem"></a>01. Dataset & Problem

- Balanced dataset of ~27K headlines labeled sarcastic vs non-sarcastic
- Headlines from satirical outlets (e.g., The Onion) vs non-satirical (e.g., HuffPost)
- Objective: detect sarcasm using linguistic features rather than site identity

---

## <a name="data-cleaning"></a>02. Data Cleaning Pipeline

Text normalization pipeline to prepare tokens for BoW/TF‑IDF and semantics methods:
- Tokenization
- Lowercasing
- Punctuation removal
- Non-alphabetic filtering
- Stopword removal
- Lemmatization

---

## <a name="feature-engineering"></a>03. Feature Engineering

- Bag of Words / TF‑IDF representations for vocabulary-driven signals
- Distributional semantics: Word2Vec, GloVe
- Contextual embeddings (reference only): BERT

Note: For this dataset size and style, sparse lexical features (BoW/TF‑IDF) were most informative.

---

## <a name="modeling"></a>04. Modeling & Evaluation

- Train/test split: 80/20
- Model: Logistic Regression with 5‑fold cross‑validation
- Cross‑validated train accuracy: ~83.7%
- Test accuracy: ~78.4%

These results are solid for short, context-light headlines where sarcasm cues are subtle.

---

## <a name="feature-importance"></a>05. Interpreting Features

Coefficient magnitudes from logistic regression highlight telltale cues:
- High-weight sarcastic indicators included terms like "area", "man", "clearly", "introduces", "announces", and profanity used for exaggeration.
- These align with satirical patterns (e.g., recurring "Area Man" trope) and sarcastic emphasis.

Permutation/drop-column importance and p-values can further validate influence beyond raw coefficients.

---

## <a name="tech-stack"></a>06. Technical Stack

- Python, Pandas, NumPy
- NLTK for tokenization/stopwords/lemmatization
- Scikit-learn for TF‑IDF and Logistic Regression (+ CV)
- Optional: Gensim (Word2Vec), Hugging Face (BERT) for extensions

---

## <a name="links"></a>07. Project Links

- **Project write-up (reference)**: [Sarcasm Detection in News Headlines](https://hi5sahil.github.io/Sarcasm-Detection-in-News-Headlines/)

---

## 🚀 Why This Project Matters

Showcases practical NLP with interpretable models that surface linguistic markers of sarcasm. Useful for content moderation, social media analytics, and news understanding, with clear paths to scale up using contextual embeddings.


