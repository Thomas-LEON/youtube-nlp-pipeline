# Large-Scale NLP Pipeline: Semantic Classification of 14M YouTube Comments 📊

> **Data Engineering & Deep Learning Research Project**  
> *End-to-end pipeline for extracting, cleaning, and classifying massive unstructured datasets using RoBERTa and mBERT.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co/)
[![Research](https://img.shields.io/badge/Status-Research_Completed-success)](#)

---

## 🎯 Executive Summary

This repository contains the methodology and architecture for a large-scale data ingestion and NLP classification project. The goal was to collect, filter, and semantically classify over **14 million YouTube comments** to analyze public perception of e-cigarettes in France.

This project demonstrates the ability to manage the full lifecycle of a machine learning project:
1. **Massive Data Ingestion** via APIs.
2. **Data Cleaning & Language Filtering** using advanced NLP models.
3. **Statistical Annotation & Inter-rater Reliability** (Fleiss' Kappa).
4. **Deep Learning Fine-Tuning** (Multilingual BERT) for semantic classification.

📄 **[Read the Full Research Paper](research_paper.pdf)** included in this repository.

---

## 🏗️ Pipeline Architecture

The processing pipeline is divided into three major stages:

### 1. Ingestion & Filtering
- Scraped 14M+ comments from ~500k videos using the YouTube Data API.
- Implemented heuristic and NLP-based filters to remove duplicates, bots, and irrelevant content.
- Utilized a zero-shot classification model (RoBERTa-based) to ensure comments were topically relevant to e-cigarettes.
- Applied language detection models to isolate French-speaking demographics.

### 2. Annotation & Ground Truth Validation
- Built a stratified sample of ~4,000 comments for manual labeling (Favorable, Unfavorable, Neutral).
- Validated annotation quality using **Fleiss’ Kappa**, ensuring statistical reliability across multiple annotators before training.

### 3. Deep Learning Classification
- Fine-tuned a **multilingual BERT (mBERT)** model for sequence classification.
- Addressed complex linguistic challenges typical of social media: sarcasm, idiomatic expressions, and heavy class imbalance.
- Evaluated performance rigorously using bootstrap confidence intervals (95% CI) for Precision, Recall, and F1-Score.

---

## 📈 Model Performance & Metrics

The fine-tuned mBERT model achieved robust performance, particularly on the complex task of identifying polarized opinions in noisy social media text.

| Class | Precision (95% CI) | Recall (95% CI) | F1-Score (95% CI) |
|---|---|---|---|
| **Class 1 (Favorable)** | 0.96 *(0.92 - 0.98)* | 0.82 *(0.77 - 0.87)* | 0.88 *(0.85 - 0.91)* |
| **Class 2 (Unfavorable)** | 0.68 *(0.60 - 0.76)* | 0.94 *(0.89 - 0.98)* | 0.79 *(0.73 - 0.84)* |

*(Note: Class 0 represents neutral/unrelated content removed during final balanced evaluation).*

---

## 📊 Exploratory Data Analysis (EDA)

Post-classification, extensive statistical analysis was conducted to characterize engagement patterns and community dynamics. 

### Engagement Distributions (CDFs)
We observed heavy-tailed distributions typical of coordinated social media behavior. 

<div align="center">
  <img src="assets/cdf_favorable_users.png" width="30%" alt="CDF Favorable Users">
  <img src="assets/cdf_channel_comments.png" width="30%" alt="CDF Channel Comments">
  <img src="assets/cdf_replies.png" width="30%" alt="CDF Replies">
</div>

*Figures: Cumulative Distribution Functions (CDFs) illustrating the concentration of comments, channel activity, and reply generation.*

---

## 🛠️ Tech Stack

- **Data Ingestion:** YouTube Data API v3, `requests`, `pandas`
- **NLP & Deep Learning:** `transformers` (HuggingFace), `torch`, `scikit-learn`
- **Visualization:** `matplotlib`, `seaborn`, `wordcloud`
- **Statistics:** `scipy`, Bootstrap Confidence Intervals

---

## 🔒 Note on Data Privacy

Due to GitHub file size limits and GDPR/Data Privacy compliance regarding user-generated content, the raw dataset (`14M rows`) and the cleaned CSV files are **not** included in this repository. 

This repository serves to showcase the code architecture (see the `notebooks/` directory for the 5-step modular pipeline), the analytical methodology, and the final statistical findings.
