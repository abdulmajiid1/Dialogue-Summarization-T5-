# Dialogue-Summarization-T5-

## 📌 Project Overview

This project focuses on **Abstractive Dialogue Summarization** using a parameter-efficient fine-tuning (PEFT) approach. We fine-tuned the `google/flan-t5-base` model on the **SAMSum corpus** using **QLoRA** (Quantized Low-Rank Adaptation). This approach allows for high-performance summarization while maintaining low computational costs.

## 🏗️ Project Structure

```bash
Dialogue-Summarization-T5/
├── src/                      # Source code for training and inference
│   ├── app.py                # Interactive Gradio web application (Demo)
│   ├── config.py             # Global hyperparameters and file paths
│   ├── model_loader.py       # 4-bit Quantization (NF4) & LoRA configuration
│   └── train.py              # Main training script (Hugging Face Trainer)
├── Final_Project_Fixed.ipynb # Comprehensive Project Notebook
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation

```

## 🚀 Quick Start

### 1. Initial Setup

First, clone the repository and enter the directory:

```bash
git clone https://github.com/abdulmajiid1/Dialogue-Summarization-T5-.git
cd Dialogue-Summarization-T5-

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Training (Optional)

If you want to retrain the model or reproduce the fine-tuning results, run the training script:

```bash
python src/train.py

```

### 4. Interactive Demo

To launch the Gradio web interface for real-time summarization, simply run the app script:

```bash
python src/app.py

```

## 🧪 Model & Implementation

1. **Base Model:** `google/flan-t5-base` (248M Parameters).
2. **Quantization:** **BitsAndBytes** 4-bit Normal Float (NF4).
3. **PEFT Strategy:** **QLoRA** (Quantized Low-Rank Adaptation).
4. **Dataset:** **SAMSum Corpus** (14k dialogues).

## 🛠️ Tech Stack

* **Core:** Python 3.10
* **DL Framework:** PyTorch, Hugging Face Transformers
* **Optimization:** PEFT, BitsAndBytes, LoRA
* **Interface:** Gradio

## 📝 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

This video explains the basics of GitHub Markdown so you can see exactly how to keep text outside of code blocks for a professional look.

