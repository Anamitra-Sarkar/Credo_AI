# 🧠 Credo AI - Next-Generation Fact-Checking Platform

**➡️ [View the Live Demo on Hugging Face Spaces](https://huggingface.co/spaces/Arko007/Credo_AI)**

## 🚀 Overview

Credo AI is a next-generation platform for combating misinformation, built on a unique **"Two-Brain" AI architecture**. It provides both immediate, high-confidence verdicts and deep, nuanced analysis for any piece of text or news article.

## ✨ Key Features

- 🤖 **Dual-AI Architecture**: Utilizes two specialized DeBERTa-v3-large models for unparalleled analysis
- ⚡ **Brain 2 (The Specialist)**: A 99.9% accurate binary classifier for rapid FAKE/REAL verdicts
- 🎯 **Brain 1 (The Nuance Expert)**: A state-of-the-art 6-class classifier for deep, nuanced analysis of complex claims
- 🌐 **Live URL Analysis**: Scrapes and analyzes content directly from news article URLs
- 💬 **Gemini Intelligence Layer**: Integrates Google's Gemini API to provide clear, conversational summaries of the AI's findings
- 📱 **Multi-Page Platform**: A polished Streamlit UI with live analysis, persistent session history, and detailed explanations of the AI

## 📁 File Structure

This repository contains the final, production-grade code for the Credo AI platform.

- **`app.py`**: The complete, self-contained Streamlit web application. This is the file deployed on Hugging Face Spaces
- **`requirements.txt`**: A list of all Python dependencies required to run the application and training scripts

### 🏋️ Training Scripts

The following scripts represent the final, battle-tested code used to train our two champion AI models on high-performance GPUs (NVIDIA A100/H100):

- **`train_brain2_A100_final_boss.py`**: The definitive script for training our "Specialist" model. It combines three massive datasets, performs intelligent deduplication, and achieves 99.9%+ accuracy
- **`train_brain1_A100_final_boss.py`**: The definitive script for training our "Nuance Expert." It combines four different LIAR datasets and uses an advanced class-weighting strategy to achieve state-of-the-art performance on this notoriously difficult benchmark
- **`refine_brain2_final.py`**: The surgical calibration script used to give our "Specialist" its final "masterclass." This script fine-tunes the model on a perfectly balanced dataset of pristine facts and high-quality fake news, correcting for real-world biases

## 💻 How to Run Locally

### Prerequisites
Make sure you have [git-lfs](https://git-lfs.com) installed:

```bash
git lfs install
```

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://huggingface.co/spaces/Arko007/Credo_AI
   cd Credo_AI
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API Key:**
   - Create a `.streamlit/secrets.toml` file in the project root
   - Add your Gemini API key to it:
     ```toml
     GOOGLE_API_KEY = "your_api_key_here"
     ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 🎯 About This Project

This project represents a powerful proof-of-concept for the future of automated fact-checking, demonstrating not only state-of-the-art performance but also a deep, real-world understanding of the challenges and limitations of modern AI.
