# 📊 YouTube NLP Pipeline

> **Large-Scale Data Engineering & Deep Learning Research Pipeline.**

> [!NOTE]
> **Research Artifact** — This repository contains the code architecture and statistical findings of an international research project analyzing 14M+ YouTube comments. Due to GDPR constraints, raw data is excluded, but the 5-step modular pipeline is available for review.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![Research](https://img.shields.io/badge/Status-Research_Completed-success)](#)

---

## 🤔 The Challenge

Analyzing public sentiment on controversial topics (like e-cigarettes) requires moving beyond basic keyword matching. Social media data is exceptionally noisy, multi-lingual, and riddled with sarcasm. 

To achieve scientific rigor, we needed an automated pipeline capable of:
1. **Ingesting** millions of unstructured comments.
2. **Filtering** out irrelevant noise using zero-shot semantic models.
3. **Classifying** nuanced stances (Favorable vs. Unfavorable) using fine-tuned Deep Learning.

---

## 🏗️ Pipeline Architecture

The solution is divided into a 5-step modular pipeline (available in the `notebooks/` directory).

```mermaid
flowchart TD
    A["🌐 YouTube Data API"] -->|"14M+ Comments"| B{"🔍 Step 1: DataPrep\nHeuristic & Zero-Shot"}
    B -->|"Filtered & Clean"| C["📊 Step 2: Agreement Calculator\nFleiss' Kappa Validation"]
    B --> D["📈 Step 3: EDA\nDistribution Analysis"]
    C -->|"Gold Standard Dataset"| E{"🧠 Step 4: Stance Detection\nmBERT Fine-Tuning"}
    E -->|"Classification"| F["🎯 Step 5: Model Characterization\nMetrics & Bootstrapping"]
    
    style A fill:#ff4757,color:#fff
    style B fill:#3742fa,color:#fff
    style C fill:#2ed573,color:#fff
    style D fill:#ffa502,color:#fff
    style E fill:#8e44ad,color:#fff
    style F fill:#2f3542,color:#fff
```

---

## ✅ What It Does (The Modules)

*   **`1_DataPrep.ipynb`** : Ingests raw data, applies deduplication, language detection, and uses RoBERTa to filter out topically irrelevant content.
*   **`2_Agreement_Calculator.ipynb`** : Calculates Fleiss’ Kappa on manually annotated samples to guarantee inter-annotator reliability before training.
*   **`3_Data_Characterization.ipynb`** : Generates statistical distributions and WordClouds to understand the dataset organically.
*   **`4_Stance_Detection_Model.ipynb`** : Fine-tunes a **multilingual BERT (mBERT)** model specifically on the polarized social media dataset.
*   **`5_Model_Characterization.ipynb`** : Evaluates model robustness using precision, recall, F1-scores, and 95% bootstrap confidence intervals.

---

## 📈 Model Performance

The fine-tuned mBERT model achieved robust performance, successfully navigating sarcasm and class imbalance.

| Class | Precision (95% CI) | Recall (95% CI) | F1-Score (95% CI) |
|---|---|---|---|
| **Favorable** | 0.96 *(0.92 - 0.98)* | 0.82 *(0.77 - 0.87)* | 0.88 *(0.85 - 0.91)* |
| **Unfavorable** | 0.68 *(0.60 - 0.76)* | 0.94 *(0.89 - 0.98)* | 0.79 *(0.73 - 0.84)* |

---

## 📊 Engagement Analytics (EDA)

We discovered heavy-tailed distributions typical of coordinated social media behavior.

<div align="center">
  <img src="assets/cdf_favorable_users.png" width="30%" alt="CDF Favorable Users">
  <img src="assets/cdf_channel_comments.png" width="30%" alt="CDF Channel Comments">
  <img src="assets/cdf_replies.png" width="30%" alt="CDF Replies">
</div>

*Figures: Cumulative Distribution Functions (CDFs) illustrating the concentration of comments, channel activity, and reply generation.*

---

## 📄 Scientific Publication

The methodology and findings of this pipeline contributed directly to an international comparative research project.

> **📄 Scientific paper currently under peer-review (Available upon request)**

---

## 🛠️ Tech Stack

| Domain | Tools Used |
|---|---|
| **Data Ingestion** | YouTube Data API v3, `requests`, `pandas` |
| **Deep Learning / NLP** | `transformers` (HuggingFace), `torch`, `scikit-learn` |
| **Data Visualization** | `matplotlib`, `seaborn`, `wordcloud` |
| **Statistics** | `scipy`, Bootstrap Confidence Intervals, Fleiss' Kappa |
