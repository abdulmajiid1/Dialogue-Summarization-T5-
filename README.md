
# 🗣️ Dialogue Summarization with Flan-T5 & QLoRA

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=for-the-badge&logo=huggingface&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-App-red?style=for-the-badge&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

## 📌 Project Overview
This project focuses on **Abstractive Dialogue Summarization** using a parameter-efficient fine-tuning (PEFT) approach. We fine-tuned the `google/flan-t5-base` model on the **SAMSum corpus** using **QLoRA** (Quantized Low-Rank Adaptation). This approach allows for high-performance summarization while maintaining low computational costs.

---

## 📂 Repository Structure
The project follows a modular architecture for reproducibility and clean code organization:

```text
Dialogue-Summarization-T5/
├── src/
│   ├── app.py                # Interactive Gradio web application (Demo)
│   ├── config.py             # Global hyperparameters and file paths
│   ├── model_loader.py       # 4-bit Quantization (NF4) & LoRA configuration
│   └── train.py              # Main training script (Hugging Face Trainer)
├── Final_Project_Fixed.ipynb # Full Project Notebook (Analysis, Training, Eval)
├── requirements.txt          # List of project dependencies
└── README.md                 # Project Documentation


## 🚀 Technical Implementation
Model Architecture
Base Model: google/flan-t5-base (248M Parameters)

Task: Abstractive Summarization (Seq2Seq)

Optimization & Fine-Tuning
Quantization: BitsAndBytes 4-bit Normal Float (NF4) for memory efficiency.

PEFT (LoRA): Rank=16, Alpha=32, Dropout=0.05.

Target Modules: Fine-tuned q (Query) and v (Value) attention layers.

##Dataset
Source: SAMSum Corpus (14k dialogues).

Preprocessing: Tokenized with T5Tokenizer and dynamic padding.
