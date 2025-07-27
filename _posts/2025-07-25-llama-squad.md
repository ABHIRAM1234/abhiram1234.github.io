---
layout: post
title: "LLaMA SQuAD: Specializing a Foundation Model for High-Stakes Question Answering"
image: "img/posts/llama_squad.png" 
tags: [LLM, Fine-Tuning, PyTorch, QLoRA, MLOps, NLP]
---

# Table of Contents
- [00. Project Overview](#overview-main)
- [01. The Problem: Hallucination in Enterprise AI](#problem)
- [02. The Solution: Specialization with QLoRA on a Rigorous Benchmark](#solution)
- [03. My Approach & Key Technical Decisions](#approach)
- [04. Quantifiable Results & Impact](#results)
- [05. Conclusion & Key Takeaways](#conclusion)
- [06. Explore the Project](#repo-link)

___

### <a name="overview-main"></a> 00. Project Overview

This project demonstrates an end-to-end workflow for specializing a general-purpose foundation model (Meta's Llama 2 & 3) for the high-stakes, extractive Question-Answering task on the SQuAD 2.0 benchmark. By leveraging memory-efficient fine-tuning (QLoRA) and a custom training strategy in PyTorch, I developed a 7B-parameter model that not only achieved a **~300% performance lift over its baseline** but also dramatically reduced model hallucination, a critical requirement for deploying reliable AI in enterprise environments.

___

### <a name="problem"></a> 01. The Problem: Hallucination in Enterprise AI

General-purpose Large Language Models (LLMs) like Llama or GPT are incredibly powerful but have a significant flaw for enterprise use: they are prone to **hallucination**. When faced with a question they cannot answer from a given context, their default behavior is to provide a plausible-sounding but incorrect answer. This makes them unreliable for tasks that demand factual accuracy, such as providing answers to a user from a specific knowledge base. The core business challenge is to create a model that is not only accurate when an answer exists but is also "honest" enough to abstain when one does not.

___

### <a name="solution"></a> 02. The Solution: Specialization with QLoRA on a Rigorous Benchmark

The goal was to specialize a compact, open-source model (Llama 2, 7B) to perform extractive QA with high fidelity. The **SQuAD 2.0 dataset** was chosen as the training and evaluation benchmark specifically because it contains over 50,000 "unanswerable" questions, making it the industry standard for testing a model's ability to avoid hallucination. The project involved fine-tuning the model to adhere to two strict rules:
1.  If an answer exists in the context, extract it verbatim.
2.  If an answer does not exist, explicitly state that it is unanswerable.

___

### <a name="approach"></a> 03. My Approach & Key Technical Decisions

To achieve this, I engineered a complete training and evaluation pipeline with several key technical components:

*   **Memory-Efficient Fine-Tuning:** I used **QLoRA (Quantized Low-Rank Adaptation)** and the `bitsandbytes` library to quantize the 7-billion-parameter Llama 2 model to 4-bit precision. This drastically reduced the memory footprint, enabling the entire fine-tuning process to run on a **single consumer-grade GPU**—a significant cost and resource saving.

*   **Masked Causal Language Modeling:** A key innovation was implementing a custom `DataCollator` in PyTorch. During training, this collator applies the cross-entropy loss function *only* to the tokens that form the JSON answer. All other tokens in the prompt (system instructions, context, question) are masked with an ID of -100. This crucial step teaches the model to focus its learning on producing the correct output format while preserving its general reasoning and chat capabilities, preventing catastrophic forgetting.

*   **Technology Stack:**
    *   **Modeling:** PyTorch, Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning)
    *   **Training:** TRL (Transformer Reinforcement Learning) for the SFTTrainer
    *   **Quantization:** `bitsandbytes` for QLoRA implementation

___

### <a name="results"></a> 04. Quantifiable Results & Impact

The fine-tuned model was rigorously evaluated against the SQuAD 2.0 validation set, showing dramatic improvements over the baseline and surpassing much larger, general-purpose models.

*   **Performance vs. Baseline:** The fine-tuned Llama 2 model achieved a **74.99% Exact Match** score, representing a **~300% improvement** over the 18.76% score of the base model.
*   **Hallucination Reduction:** Crucially, the model's ability to correctly abstain on unanswerable questions **skyrocketed from a near-unusable 3.72% in the base model to 82.02%**, demonstrating a significant reduction in hallucination.
*   **Performance vs. Industry Models:** The specialized 7B model's performance **surpassed that of the much larger Llama 2 70B model** and even OpenAI's GPT-4 on this specific, structured task.

___

### <a name="conclusion"></a> 05. Conclusion & Key Takeaways

This project successfully demonstrates that by combining modern, memory-efficient fine-tuning techniques with a targeted training strategy, a compact, open-source LLM can be specialized to outperform even massive, general-purpose models on specific enterprise tasks. More importantly, it proves that a model can be explicitly trained for reliability and honesty, overcoming the critical issue of hallucination and making it a viable candidate for production systems.

___

### <a name="repo-link"></a> 06. Explore the Project

The complete code, training scripts, and detailed documentation are available on my GitHub.

[**View on GitHub: ABHIRAM1234/llama-squad**](https://github.com/ABHIRAM1234/llama-squad)