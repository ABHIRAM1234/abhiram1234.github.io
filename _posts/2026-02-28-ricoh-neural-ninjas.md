---
layout: post
title: "RicohLibrary: Agentic RAG System for Technical Support"
image: "/posts/ricoh-img.png"
tags: [RAG, LangGraph, NLP, Python, ChromaDB, Streamlit, Agentic AI]
summary: "Agentic AI technical support system using LangGraph, hybrid retrieval (ChromaDB + BM25 + RRF), and Claude for grounded, cited answers from Ricoh product manuals. Built at HackVerse 2026."
order: 3
github_url: "https://github.com/ABHIRAM1234/Ricoh_Neural_Ninjas"
---

# Table of Contents
- [00. Project Overview](#project-overview)
- [01. Problem Statement](#problem-statement)
- [02. Solution Overview](#solution-overview)
- [03. Architecture & System Design](#architecture)
- [04. Data Handling & Preprocessing](#data-handling)
- [05. Modeling & AI Strategy](#modeling)
- [06. Evaluation & Results](#evaluation)
- [07. Tech Stack](#tech-stack)
- [08. Project Links](#links)

---

## <a name="project-overview"></a>00. Project Overview

**What this project is (in plain English):**
I built an **agentic AI technical support system** called **RicohLibrary** that answers complex, multi-part questions about Ricoh products using only provided documentation. The system ingests PDF manuals, retrieves relevant passages via **hybrid search** (semantic + keyword), reasons through a **LangGraph state machine** with a verify-and-retry loop, and generates accurate, **cited answers** — refusing to answer when evidence is insufficient. Built during **HackVerse 2026** hackathon.

**At a glance**
- **Problem:** Field technicians waste significant time searching through hundreds of pages of Ricoh product manuals to find specific procedures, error codes, and configuration steps.
- **Solution:** An intelligent agentic system that ingests manuals, understands natural-language questions, retrieves relevant passages, and generates accurate, cited answers with zero hallucination.
- **Code:** [GitHub Repository](https://github.com/ABHIRAM1234/Ricoh_Neural_Ninjas)

---

## <a name="problem-statement"></a>01. Problem Statement

Field technicians and support engineers waste significant time searching through hundreds of pages of Ricoh product manuals to find specific procedures, error code resolutions, and configuration steps. We needed an intelligent system that can:
1. **Ingest** PDF manuals and preserve metadata
2. **Understand** natural-language questions (including multi-part queries)
3. **Retrieve** the most relevant passages using hybrid search
4. **Generate** accurate, cited answers with strict hallucination control

---

## <a name="solution-overview"></a>02. Solution Overview

**RicohLibrary** is an agentic AI system that:

1. **Ingests** Ricoh PDF manuals using PyMuPDF with metadata-preserving chunking (500 words, 50-word overlap)
2. **Retrieves** relevant passages via a **hybrid engine** combining semantic vector search (ChromaDB + MiniLM) and keyword search (BM25), fused with Reciprocal Rank Fusion
3. **Reasons** through a LangGraph state machine with 4 nodes (Planner → Retriever → Verifier → Synthesizer) and a conditional retry loop
4. **Generates** grounded answers with strict `[Document Name, Page X]` citations
5. **Visualises** the full reasoning process in a "Glass Box" Streamlit dashboard
6. **Polyglot Support:** Automatically detects user language and answers in that language while preserving English citations

---

## <a name="architecture"></a>03. Architecture & System Design

The system uses a **4-node LangGraph state machine**:

```
User Question
     ↓
┌──────────────────────────────────────────────────┐
│           LangGraph State Machine                 │
│                                                   │
│  🧠 PLANNER → 📚 RETRIEVER (2-pass) → ✅ VERIFIER│
│     ↑          │ Pass 1: sub-queries     │       │
│     │          │ Pass 2: entity-boosted  │       │
│     └── (INSUFFICIENT & iter < 2) ───────┘       │
│                               │                   │
│                       (SUFFICIENT)                │
│                               ↓                   │
│                       💬 SYNTHESIZER              │
└──────────────────────────────────────────────────┘
```

**Why this architecture?**
- **Hybrid retrieval** because pure vector search misses exact matches on error codes and model numbers, while pure BM25 misses semantic similarity
- **RRF fusion** because BM25 and cosine similarity scores are incommensurable — rank-based fusion avoids score normalisation issues
- **Agentic loop** because single-pass retrieval often misses evidence for multi-part questions

---

## <a name="data-handling"></a>04. Data Handling & Preprocessing

- **Dataset:** Official Ricoh product documentation PDFs
- **Extraction:** PyMuPDF extracts raw text page-by-page, preserving `source_document` and `page_number` metadata
- **Chunking:** Sliding window of ~500 words with 50-word overlap (word-based for semantic coherence)
- **Storage:** ChromaDB (vector index) + pickled BM25 (keyword index), both persisted for fast restarts

---

## <a name="modeling"></a>05. Modeling & AI Strategy

### LLM: Claude Sonnet (Anthropic)
- Temperature=0.0 for deterministic, factual answers
- 4 specialized prompts: Planner, Verifier, Synthesizer, and Retry Context

### Retrieval Strategy
- **Semantic:** ChromaDB with local all-MiniLM-L6-v2 embeddings
- **Keyword:** BM25Okapi over full chunk corpus
- **Fusion:** RRF(k=60) merges rank positions, returning top-5 fused results per sub-query

### Hallucination Control
- Synthesizer instructed to say "Information unavailable in provided documents" when evidence is insufficient

---

## <a name="evaluation"></a>06. Evaluation & Results

Evaluated on **10 official hackathon test questions**:

- **Average latency:** ~13.9s per question
- **Citation accuracy:** Every answer includes traceable `[Document Name, Page X]` citations
- **Hallucination control:** Agent correctly refused to fabricate answers when evidence was missing
- **Retrieval coverage:** BM25 + vector search consistently returned 5+ relevant chunks per sub-query

**Business Impact:**
- Estimated **60%+ reduction** in resolution time for support engineers
- Glass Box transparency lets supervisors verify answer quality
- Modular architecture allows swapping LLM providers via a single config change

---

## <a name="tech-stack"></a>07. Tech Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.11 |
| PDF Parsing | PyMuPDF |
| Vector Database | ChromaDB (all-MiniLM-L6-v2) |
| Keyword Search | BM25Okapi |
| Agentic Framework | LangGraph |
| LLM | Claude Sonnet (Anthropic) |
| UI | Streamlit |

---

## <a name="links"></a>08. Project Links

- **GitHub:** [ABHIRAM1234/Ricoh_Neural_Ninjas](https://github.com/ABHIRAM1234/Ricoh_Neural_Ninjas)
