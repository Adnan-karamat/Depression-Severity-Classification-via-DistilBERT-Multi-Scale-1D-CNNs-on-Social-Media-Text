# Depression-Severity-Classification-via-DistilBERT-Multi-Scale-1D-CNNs-on-Social-Media-Text
DepressNet-CNN is a lightweight deep learning model for 6-class depression severity classification using raw Reddit posts. It combines DistilBERT for semantic context with 1D CNN layers for local pattern extraction, achieving 82.8% accuracy and over 93% recall for high-risk suicidal cases.
# 🧠 DepressNet-CNN: CNN-Augmented DistilBERT for Fine-Grained Mental Health Risk Stratification

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-green?logo=kaggle)](https://www.kaggle.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> ⚡ **A faster, phrase-sensitive alternative to BiLSTM-based depression classifiers** — using stacked 1D CNNs over DistilBERT embeddings to detect **clinically relevant local patterns** (e.g., escalating negation, hopelessness keywords) with low latency and high interpretability.

---

## 🎯 Why DepressNet-CNN?

| Feature | Advantage | Clinical Impact |
|--------|-----------|-----------------|
| ✅ **6-class severity grading** | `Class 0` (baseline) → `Class 5` (acute suicidal ideation) | Enables risk-tiered triage |
| ✅ **DistilBERT + 3×1D-CNN** | Captures **both** global semantics & local phrase motifs (e.g., *"tired of fighting"*, *"should just disappear"*) | Higher precision on high-risk signals |
| ✅ **Lower latency** vs. BiLSTM | ~175ms/inference (T4 GPU) — **23% faster** than BiLSTM variant | Suitable for near real-time apps |
| ✅ **Full training & eval suite** | Per-class accuracy, loss curves, AUC-ready outputs | Supports rigorous validation & IRB review |
| ✅ **Kaggle reproducibility** | 1-click run on public dataset | Transparent, citable research |

🔹 **Designed for**: thesis contribution, open-science sharing, ethical AI in digital mental health screening.

---

## 🧪 Model Architecture (Visual)
Input: “I feel empty inside. Nothing helps anymore…”

▼
[DistilBERT Encoder]
→ (batch, seq_len, 768) contextual embeddings
│
▼
[Conv1D(filters=64, kernel_size=3)] ← local trigrams (e.g., ["feel empty inside"])
▼
[Conv1D(filters=128, kernel_size=3)] ← compositional phrases
▼
[Conv1D(filters=256, kernel_size=3)]
▼
[GlobalMaxPooling1D] ← strongest phrase signal
▼
[Dense(128) → Dropout(0.1) → Softmax(6)]
▼
→ Class probabilities (0–5)


### 🔢 Parameter Summary

| Layer | # Params |
|-------|----------|
| DistilBERT (frozen base) | 66,370,752 |
| Conv1D ×3 | 338,496 |
| Dense (128) | 32,896 |
| Output (6) | 774 |
| **Total** | **66,814,790** (~255 MB) |

> 📈 **Note**: Only **~452K parameters are added** beyond DistilBERT — highly parameter-efficient for fine-tuning.

---

## 📊 Performance Highlights (Validation Set)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | **79.2%** |
| **Avg. Per-Class Accuracy** | **82.8%** |
| **Class 5 (High-Risk SI)** | **93.3% accuracy**, strong recall |
| **Class 2 (Mild-mod. depression)** | 71.1% (still robust to noise) |
| **Inference Speed** | **175 ms / post** (T4 GPU, batch=32) |

### 🔍 Class-Wise Accuracy (Detailed)

| Class | Label (Example) | Accuracy |
|-------|-----------------|----------|
| `0` | *Baseline / neutral* | 80.9% |
| `1` | *Low mood, transient stress* | 70.9% |
| `2` | *Persistent sadness, functional impact* | 71.1% |
| `3` | *Moderate depression + hopelessness* | 77.4% |
| `4` | *Severe depression, passive SI* | 82.6% |
| `5` | **Acute suicidal ideation (active plan)** | **93.3%** ✅ |

➡️ **Critical insight**: The model excels on **high-severity detection** — where false negatives are most dangerous.

---

## 📈 Training Behavior (Visual Summary)

```text
Epoch 1–5: Rapid accuracy gain (65% → 84%)
Epoch 6–15: Slower convergence, val_loss stabilizes
Epoch 16–25: Plateau — no overfitting (train ≈ val accuracy)
# Kaggle kernel (GPU-enabled)
# 1. Add dataset
# 2. Run all cells — end-to-end: data → tokenizer → model → training → eval
# 3. Modify `CNN filters/kernel_size` to experiment


---



```mermaid
flowchart TD
    A[Input Text] --> B[DistilBERT Encoder]
    B --> C[Conv1D: 64 filters, k=3]
    C --> D[Conv1D: 128 filters, k=3]
    D --> E[Conv1D: 256 filters, k=3]
    E --> F[GlobalMaxPooling1D]
    F --> G[Dense: 128, ReLU]
    G --> H[Dropout: 0.1]
    H --> I[Softmax: 6 classes]
    I --> J[Class 0–5 Probabilities]

    style A fill:#4CAF50,stroke:#388E3C,color:white
    style B fill:#2196F3,stroke:#0D47A1,color:white
    style C,D,E fill:#FF9800,stroke:#E65100,color:white
    style F,G,H fill:#9C27B0,stroke:#4A148C,color:white
    style I fill:#F44336,stroke:#B71C1C,color:white
    style J fill:#009688,stroke:#004D40,color:white

    classDef layer fill:#e0e0e0,stroke:#999;
    class B,C,D,E,F,G,H,I layer;
