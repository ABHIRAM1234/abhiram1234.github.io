---
layout: post
title: "Amazon Reviews NLP Analysis: Sentiment, Topics & LLM-Powered Recommendations"
image: "/posts/streamlit-app-preview-img.png"
tags: [NLP, Sentiment Analysis, Topic Modeling, Python, VADER, LDA, Streamlit, LangChain, Classification]
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Why This Problem Matters](#why-this-matters)
- [02. Key Concepts: NLP Tools Used](#concepts)
- [03. My Step-by-Step Thought Process](#thought-process)
- [04. Data & Product](#data)
- [05. Methods in Detail](#methods)
- [06. Key Insights](#insights)
- [07. Limitations & Future Work](#limitations)
- [08. Technical Stack](#stack)
- [09. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

**What this project is (in plain English):**  
I analyzed **customer reviews** for a single Amazon product—**"Miracle Noodle Zero Carb, Gluten Free Shirataki Pasta, Spinach Angel Hair"**—to answer: (1) **What do customers love or hate?** (2) **What themes show up in positive vs negative reviews?** and (3) **Can we automatically label a review as high or low sentiment?** I used **Natural Language Processing (NLP)** for sentiment and topic discovery, **Machine Learning** for classification, and built a **Streamlit dashboard** with **LangChain** so stakeholders can explore the results and get LLM-powered recommendations.

**At a glance**
- **Data:** Amazon Reviews (Jianmo et al., 2019), Grocery & Gourmet Food subset → filtered to verified purchases, then one product (Miracle Noodle) chosen for deep analysis
- **NLP:** Text preprocessing, **VADER** sentiment analysis, **LDA** topic modeling
- **ML:** Binary review classification (high vs low sentiment); best model: **Logistic Regression + TF-IDF** (~96% test accuracy)
- **App:** [Streamlit dashboard](https://amazon-reviews-nlp-analysis.streamlit.app/) with Plotly visualizations and LangChain-powered recommendations
- **Repo:** [ABHIRAM1234/amazon-reviews-nlp-analysis](https://github.com/ABHIRAM1234/amazon-reviews-nlp-analysis) (forked from [jirvingphd/amazon-reviews-nlp-analysis](https://github.com/jirvingphd/amazon-reviews-nlp-analysis))

---

## <a name="why-this-matters"></a>01. Why This Problem Matters

**Business context:**  
Product teams and marketers need to know *why* customers love or hate a product. Reading thousands of reviews by hand does not scale. We need a way to (1) **summarize sentiment** (positive vs negative), (2) **discover themes** (e.g. “texture,” “smell,” “health benefits”) without predefining them, and (3) **classify new reviews** automatically so support or product can prioritize issues.

**What we need:**  
- **Sentiment:** A consistent way to score each review’s emotional tone (positive/negative/neutral).  
- **Topics:** Unsupervised discovery of what customers talk about in high- vs low-rating reviews.  
- **Classification:** A model that predicts high vs low sentiment from text so we can triage at scale.  
- **Delivery:** An interactive app so non-technical users can explore results and get actionable recommendations (including LLM-generated ones).

**Why this product:**  
I chose a product with a **large share of 1-star reviews** (Miracle Noodle) so we could study *critical* feedback in depth—texture, smell, preparation—and balance it with positive themes (health, taste, ease of use). That makes the analysis useful for product improvement and marketing.

---

## <a name="concepts"></a>02. Key Concepts: NLP Tools Used

**VADER (Valence Aware Dictionary and sEntiment Reasoner)**  
A **lexicon-based** sentiment tool: it uses a pre-built dictionary of words and their positive/negative strength, plus rules for punctuation and negation. It outputs positive, negative, neutral, and a **compound** score (e.g. -1 to +1). It’s designed for **informal text** (reviews, social media), so it handles emoticons and casual language well. I used it to score every review and to compare sentiment to star ratings and to preprocessing choices.

**LDA (Latent Dirichlet Allocation)**  
A **topic model**: it assumes each document is a mixture of a few “topics,” and each topic is a distribution over words. We don’t label topics beforehand—LDA discovers them. I ran LDA **separately** on high-rating and low-rating reviews so we can see which themes dominate in each group (e.g. “texture,” “smell” in negative; “health,” “taste” in positive).

**TF-IDF + Logistic Regression**  
**TF-IDF** turns each review into a vector of word weights (how important each word is in the document vs in the corpus). **Logistic Regression** then learns a linear boundary between “high sentiment” and “low sentiment” from those vectors. It’s interpretable (we can inspect coefficients) and fast, and it gave ~96% test accuracy as a strong baseline before any deep learning.

**Streamlit + LangChain**  
**Streamlit** powers the interactive dashboard (plots, filters, tables). **LangChain** is used to add **LLM-based recommendations**—e.g. summarizing insights or suggesting actions from the analysis—so the app doesn’t just show charts but can generate narrative recommendations.

---

## <a name="thought-process"></a>03. My Step-by-Step Thought Process

I approached the project as: get the data → pick a product → clean and explore text → sentiment and topics → classification → app and LLM.

---

### Step 1: Get the Data and Choose One Product to Analyze in Depth

**What I did:**  
I used the **Amazon Reviews** dataset (Jianmo et al., 2019), Grocery & Gourmet Food category (millions of reviews, hundreds of thousands of products). I kept only **verified purchases** and products in the Grocery `main_cat` so we focus on real, relevant feedback.

**Why one product:**  
Doing deep NLP (preprocessing, sentiment, LDA, classification) on one product keeps the narrative clear and lets us explain every step. Scaling to many products would be a follow-up (e.g. the same pipeline per product or per category).

**How I chose the product:**  
I wanted a product with **enough 1-star and 5-star reviews** so we have clear “positive” and “negative” text. I selected the product with the **largest proportion of 1-star reviews** (Miracle Noodle) so we could study *why* customers are unhappy—texture, smell, preparation—and contrast with what lovers of the product say. This makes the analysis directly useful for product and marketing.

---

### Step 2: Preprocess Text and Explore Impact on Sentiment

**What I did:**  
I preprocessed review text with **tokenization** (split into words), **lemmatization** (reduce words to base form, e.g. “running” → “run”), and **stopword removal** (drop common words like “the,” “and”). I ran **VADER** on both raw and preprocessed text and compared sentiment scores to star ratings.

**Why this matters:**  
Preprocessing **changes** sentiment scores. For example, lemmatization can shift the compound score because VADER’s lexicon may list different forms of a word. I documented this so anyone reproducing the analysis knows that **preprocessing choices affect results**—important for reproducibility and for deciding how to treat text in production.

**What I learned:**  
There is a strong correlation between star ratings and VADER scores overall; lemmatization sometimes reduced “negative” strength in low-rating reviews. So we have to choose preprocessing deliberately and report it clearly.

---

### Step 3: Discover Themes with LDA (Separately for High vs Low Ratings)

**What I did:**  
I split reviews into **high-rating** (e.g. 4–5 stars) and **low-rating** (e.g. 1–2 stars), then ran **LDA** on each set. I inspected **top terms per topic** and (where available) **intertopic distance** plots to name topics (e.g. “Texture and Smell,” “Health and Taste”).

**Why separate LDA for high vs low:**  
If we ran LDA on all reviews together, we’d get mixed topics. By splitting, we see “what do happy customers talk about?” vs “what do unhappy customers talk about?”—so we get clear, actionable themes for product and marketing.

**What I learned:**  
Positive reviews emphasized health benefits, low calories, flavor, and ease of use. Negative reviews emphasized texture (rubbery, slimy), smell (fishy), taste, and preparation difficulty. These themes directly suggest where to improve the product and what to highlight in marketing.

---

### Step 4: Build a Classifier (High vs Low Sentiment)

**What I did:**  
I framed the task as **binary classification**: high sentiment (e.g. 4–5 stars or positive VADER) vs low sentiment (e.g. 1–2 stars or negative VADER). I tried several models (Dummy baseline, Naive Bayes, **Logistic Regression**, Random Forest, Linear SVC) with **TF-IDF** features. I tuned and evaluated with a proper train/test split and reported precision, recall, and accuracy.

**Why Logistic Regression + TF-IDF:**  
It’s **interpretable** (we can see which words push toward high or low sentiment), **fast**, and it achieved ~96% test accuracy—a strong baseline. Deep learning (e.g. Part-04 sequence models) can be added later for more nuance, but this gives a clear, explainable model for “is this review positive or negative?”

**What I learned:**  
The classifier generalizes well; preprocessing and label definition (e.g. threshold on stars or VADER) matter for real deployment. Documenting the label rule and preprocessing is essential for reproducibility.

---

### Step 5: Put It All in a Streamlit App and Add LLM Recommendations

**What I did:**  
I built a **Streamlit** app that loads the analysis results (or recomputes them) and shows: rating distribution, sentiment vs preprocessing, LDA topics, and classification performance. I integrated **LangChain** so the app can generate **LLM-based recommendations**—e.g. “Based on negative themes (texture, smell), consider …”—so stakeholders get both charts and narrative suggestions.

**Why an app:**  
Non-technical users (product, marketing) can explore the data and insights without opening notebooks. The LLM layer turns the analysis into actionable language, closing the loop from data to decisions.

**Repo structure:**  
Notebooks Part-00–Part-05 cover planning, data prep, NLP preprocessing, EDA, classification, deep NLP (Part-04), and Streamlit + LangChain (Part-05). The app is the “face” of the project; the notebooks are the reproducible pipeline.

---

## <a name="data"></a>04. Data & Product

- **Dataset:** Amazon Reviews (Jianmo et al., 2019), Grocery & Gourmet Food category  
  - **Scale:** 5,074,160 reviews; 287,209 products (subset used after filtering)
- **Filtering:** Verified purchases only; products under `main_cat` Grocery
- **Product selection:** Products with substantial 1- and 5-star reviews; final choice had the **largest proportion of 1-star reviews** to focus on critical feedback
- **Selected product:** *Miracle Noodle Zero Carb, Gluten Free Shirataki Pasta, Spinach Angel Hair, 7-Ounce (Pack of 24)* — Brand: Miracle Noodle; categories: Grocery & Gourmet Food, Pasta & Noodles, Noodles, Shirataki

Rating distribution and trends (e.g. average rating by year, share of 1- vs 5-star reviews) are explored in the notebooks and reflected in the Streamlit app.

---

## <a name="methods"></a>05. Methods in Detail

**Text preprocessing**  
Tokenization, lemmatization, stopword removal. I analyzed how each step affects VADER scores and classifier performance so preprocessing choices are explicit and reproducible.

**Sentiment analysis**  
VADER: positive/negative/neutral and compound scores per review. Compared to star ratings and to preprocessing (raw vs lemmatized, etc.).

**Topic modeling**  
LDA run separately on high-rating and low-rating review corpora. Number of topics chosen via coherence or interpretability; top terms and intertopic plots used to name topics (e.g. “Texture and Smell,” “Health and Taste”).

**Classification**  
Binary label (high vs low sentiment) from star rating or VADER threshold. Features: TF-IDF (unigrams and/or bigrams). Models: Dummy, Naive Bayes, Logistic Regression, Random Forest, Linear SVC. Best: **Logistic Regression + TF-IDF** — ~96% test accuracy (precision/recall ~0.95–0.97 per class). Train/test split and (where used) cross-validation reported.

**Deep NLP & app**  
Part-04: Deep NLP text prep, sequence models, PyTorch pipelines. Part-05: Streamlit (Plotly) and LangChain for LLM-based recommendations. Companion app: [Amazon Reviews NLP Analysis & Predictions](https://amazon-reviews-nlp-analysis.streamlit.app/).

---

## <a name="insights"></a>06. Key Insights

- **Positive themes (LDA):** Health benefits, low-calorie appeal, flavor, ease of use; satisfaction and “love” for the product.
- **Negative themes (LDA):** Texture (rubbery, slimy), smell (e.g. fishy), taste, preparation; words like “disgusting,” “gross,” “horrible.”
- **Sentiment vs ratings:** Strong correlation between star ratings and VADER; **preprocessing (e.g. lemmatization) materially affects sentiment scores**—critical for reproducibility and for production choices.
- **Classification:** Logistic Regression + TF-IDF gives a strong, interpretable baseline (~96% accuracy) for high/low sentiment; a good starting point before adding deep learning.

---

## <a name="limitations"></a>07. Limitations & Future Work

- **Single product:** Results are specific to Miracle Noodle / shirataki-style products; generalizing to other categories would require running the same pipeline on other products or pooling by category.
- **Lexicon and bag-of-words:** VADER and TF-IDF don’t capture full context (sarcasm, long-range dependency); Part-04 deep NLP and sequence models extend this.
- **Next steps (from repo):** HuggingFace sentiment pipelines, deeper NLP modeling, and integrating sentiment/LDA outputs more deeply into the Streamlit app (e.g. “explain this topic” or “recommend actions per segment”).

---

## <a name="stack"></a>08. Technical Stack

- **Language:** Python
- **NLP:** VADER, NLTK (or similar) for preprocessing, Gensim (or equivalent) for LDA
- **ML:** Scikit-learn (Logistic Regression, TF-IDF, Naive Bayes, Random Forest, SVC)
- **Deep NLP:** PyTorch (Part-04 notebooks)
- **App:** Streamlit, Plotly, LangChain (LLM recommendations)
- **Workflow:** Jupyter notebooks (Part-00–Part-05), `custom_functions`, `config`, `data`, `eda`, `models`, `pages` for the Streamlit app

---

## <a name="links"></a>09. Project Links

- **GitHub:** [ABHIRAM1234/amazon-reviews-nlp-analysis](https://github.com/ABHIRAM1234/amazon-reviews-nlp-analysis) — NLP analysis of Amazon Reviews with companion Streamlit app and LLM/AI-based recommendations
- **Streamlit app:** [Amazon Reviews NLP Analysis & Predictions](https://amazon-reviews-nlp-analysis.streamlit.app/)

---

## Why This Project Matters (For Recruiters)

This project shows **end-to-end NLP for product analytics**: from raw reviews to sentiment (VADER), themes (LDA), and a production-style classifier (Logistic Regression + TF-IDF), plus an interactive **Streamlit** app and **LangChain**-powered recommendations. It demonstrates that I can frame a business question, choose appropriate NLP and ML methods, implement them, document preprocessing and its impact, and deliver insights through an app—skills that apply to NLP, customer insights, and applied ML roles.
