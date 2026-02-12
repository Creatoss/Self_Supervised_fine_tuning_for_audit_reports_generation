---
title: Audit Report Generator
emoji: 📊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.19.2
app_file: app.py
pinned: false
license: mit
---

# 📊 AI Audit Report Generator

This is a specialized LLM application designed to draft professional audit report sections. It uses a **Mistral-7B** model fine-tuned on audit documentation to generate factual, structured, and compliant audit text.

## 🚀 Features

- **Instruction-Based Generation**: Ask for specific report sections (e.g., "Draft a Key Audit Matter paragraph for Revenue").
- **Context-Aware**: Paste relevant financial figures or inspection findings to ground the generation (RAG-style).
- **Professional Tone**: The model mimics the formal style of Big 4 audit reports.

## 🛠️ Setup & Local Development

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Download/Place Model**:
    Ensure your fine-tuned model (adapter or full model) is placed in `models/`.
    *   Example: `models/audit-mistral-7b-qlora/adapter_model.safetensors`

3.  **Run App**:
    ```bash
    python app.py
    ```

## 📦 Deployment

This repository is configured for deployment on **Hugging Face Spaces**.

### GitHub Actions Deployment
This project includes a `.github/workflows/huggingface_deploy.yml` workflow that automatically syncs changes from the `deployment/` folder to your Space.

**Prerequisites:**
1.  Create a Space on Hugging Face (SDK: Gradio).
2.  Get a Hugging Face **Access Token** (Write permission).
3.  In this GitHub Repo, go to **Settings > Secrets and variables > Actions**:
    *   Add Repository Secret: `HF_TOKEN` (Your HF Access Token).
    *   Add Repository Variable: `HF_SPACE_ID` (Your Space ID, e.g., `username/audit-generator`).

## 🧠 Model Details

- **Base Model**: Mistral-7B-v0.1
- **Fine-Tuning**: QLoRA (4-bit) on Audit datasets.
- **Objective**: Reduce hallucinations and maintain strict professional formatting.
