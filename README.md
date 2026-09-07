# Vernacular Historical NER (`vernacular-historical-ner`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-PEFT-blue)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Parameter-efficient Named Entity Recognition and Entity Resolution pipelines optimized for low-resource, historical, and vernacular South Asian corpora.**

---

## 📌 Abstract & Research Problem

Historical and devotional corpora across South Asia—such as colonial-era print archives, regional vernacular tracts, and Bhakti literature—present severe computational hurdles for off-the-shelf Natural Language Processing (NLP) toolkits:

1. **Orthographic Variance & Code-Switching:** Historical orthography, archaic vocabulary, and multi-lingual mixing (e.g., Braj Bhasha, Early Hindi, Urdu, and colonial English) degrade standard tokenizers.
2. **Domain-Specific Entity Taxonomies:** Standard NER taxonomies (`PER`, `LOC`, `ORG`) fail to capture institutional, theological, and socio-religious dynamics (e.g., monastic orders, sectarian titles, sacred geography, and vernacular text citations).
3. **Low-Resource Constraints:** Gold-standard annotated historical corpora in South Asian vernaculars remain sparse.

This repository provides a modular, production-ready pipeline that fine-tunes multilingual Transformer architectures (`mBERT`, `XLM-RoBERTa`) using **Parameter-Efficient Fine-Tuning (PEFT / LoRA)**. The system isolates and extracts fine-grained historical entities from unstandardized archives with minimal computational overhead.

---

## 🏗️ Architecture & Pipeline Overview

```text
               +----------------------------------+
               |  Unstructured Historical Archive |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Preprocessing & Sentence Chunking|
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Custom BIO Tag Alignment Engine  |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  mBERT / XLM-RoBERTa + LoRA PEFT |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               | Extracted Entities & Resolution  |
               +----------------------------------+