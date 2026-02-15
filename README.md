---
title: Audit Report Generator
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.15.0
python_version: "3.13"
app_file: app.py
pinned: false
license: mit
---

# 📊 Self-Supervised Fine-Tuning for Audit Report Generation

A research project demonstrating advanced LLM fine-tuning techniques to transform a general-purpose language model into a specialized audit report assistant. This work explores the complete lifecycle: **Self-Supervised Fine-Tuning (SFT)** → **Retrieval-Augmented Generation (RAG)** → **Direct Preference Optimization (DPO)**.

---

## 🎯 Project Overview

### The Challenge
After initial fine-tuning on audit report PDFs, the model exhibited a critical behavioral flaw: it began acting as a **"PDF reader"** rather than an **"audit assistant"**. The model would regurgitate memorized report fragments instead of synthesizing new, contextually appropriate responses.

**Initial Metrics (Post-SFT):**
- Perplexity: **3.2** (good language modeling)
- Cosine Similarity: **0.64** (moderate semantic relevance)
- Behavior: Hallucinating specific figures, copying boilerplate text

### The Solution
We implemented a **two-phase alignment strategy**:
1. **RAG Integration**: Grounding responses in retrieved enterprise data
2. **DPO Realignment**: Using Gemini-generated preference pairs to teach the model to prioritize context over memorized patterns

**Final Metrics (Post-DPO):**
- Cosine Similarity: **0.77** (+20% improvement)
- Faithfulness Score: **95/100** (LLM-as-a-Judge evaluation)
- Behavior: Context-aware, factually grounded responses

---

## 🏗️ Architecture

### Phase 1: Self-Supervised Fine-Tuning (SFT)
**Notebook:** `audit_model_finetuning.ipynb`

- **Objective**: Teach Mistral-7B the domain-specific language of audit reports
- **Method**: QLoRA (4-bit quantization) on 100+ audit PDFs
- **Result**: Model learned professional tone and structure but over-fitted to training data

### Phase 2: Retrieval-Augmented Generation (RAG)
**Notebook:** `rag_audit_generation.ipynb`

- **Objective**: Ground model responses in factual enterprise data
- **Components**:
  - Vector Store: ChromaDB with `sentence-transformers/all-MiniLM-L6-v2`
  - Retriever: Top-k relevant document chunks
  - LLM-as-a-Judge: Gemini 2.5 Flash for hallucination detection
- **Discovery**: Model still preferred its internal "hallucinated memory" over retrieved facts (Faithfulness: 0/100)

### Phase 3: Direct Preference Optimization (DPO)
**Notebook:** `audit_model_dpo_implementation.ipynb`

- **Objective**: Align model to prefer factual, context-grounded responses
- **Data Generation**:
  - **Rejected**: Hallucinated responses from the SFT model
  - **Chosen**: Gemini 2.5 Flash corrections with strict factual grounding
- **Training**: DPO on preference pairs to shift model behavior
- **Result**: Dramatic improvement in semantic alignment and factual accuracy

---

## 📊 Key Results

| Metric | Pre-DPO | Post-DPO | Improvement |
|--------|---------|----------|-------------|
| Cosine Similarity | 0.64 | 0.77 | +20% |
| Faithfulness Score | 0/100 | 95/100 | +95pts |
| Hallucination Rate | High | Minimal | ✅ |

---

## 🛠️ Setup & Local Development

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running the Notebooks
1. **Fine-Tuning**: `notebooks/audit_model_finetuning.ipynb`
2. **RAG Setup**: `notebooks/rag_audit_generation.ipynb`
3. **DPO Training**: `notebooks/audit_model_dpo_implementation.ipynb`

### Running the Gradio App
```bash
python app.py
```

---

## 📦 Deployment

This repository is configured for deployment on **Hugging Face Spaces**.

### GitHub Actions Deployment
This project includes a `.github/workflows/huggingface_deploy.yml` workflow that automatically syncs changes to your Space.

**Prerequisites:**
1. Create a Space on Hugging Face (SDK: Gradio)
2. Get a Hugging Face **Access Token** (Write permission)
3. In this GitHub Repo, go to **Settings > Secrets and variables > Actions**:
   - Add Repository Secret: `HF_TOKEN` (Your HF Access Token)
   - Add Repository Variable: `HF_SPACE_ID` (Your Space ID, e.g., `username/audit-generator`)

---

## 🧠 Technical Details

### Model Architecture
- **Base Model**: Mistral-7B-v0.1
- **Fine-Tuning Method**: QLoRA (4-bit quantization)
- **Alignment Method**: Direct Preference Optimization (DPO)
- **Deployment**: CPU-optimized with bfloat16 for 16GB RAM environments

### Evaluation Framework
- **Semantic Relevance**: Cosine similarity between query and response embeddings
- **Factual Accuracy**: LLM-as-a-Judge (Gemini 2.5 Flash) comparing response to source data
- **Perplexity**: Language modeling quality metric

### Data Pipeline
- **Training Data**: Audit inspection reports (PDFs)
- **RAG Knowledge Base**: Enterprise-specific financial data
- **DPO Preference Pairs**: Generated via Gemini 2.5 Flash

---

## 📁 Project Structure

```
.
├── notebooks/
│   ├── audit_model_finetuning.ipynb      # Phase 1: SFT
│   ├── rag_audit_generation.ipynb        # Phase 2: RAG + Evaluation
│   └── audit_model_dpo_implementation.ipynb  # Phase 3: DPO
├── Models/
│   └── Self_Supervised_finetuning_Model/
│       ├── audit-mistral-7b-qlora/       # SFT checkpoint
│       └── audit-mistral-7b-dpo/         # DPO checkpoint
├── app.py                                 # Gradio deployment interface
├── requirements.txt
└── README.md
```

---

## 🔬 Research Contributions

1. **Demonstrated the "PDF Reader" Problem**: Showed how SFT alone can cause models to memorize rather than reason
2. **RAG Conflict Analysis**: Proved that fine-tuned models can ignore retrieved context in favor of hallucinated patterns
3. **DPO for Factual Alignment**: Successfully used preference optimization to realign model behavior toward context-grounding

---

## 📄 License

MIT License - See LICENSE file for details

