# Dialogue Summarization with Flan-T5 & QLoRA

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange?style=for-the-badge&logo=huggingface&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-App-red?style=for-the-badge&logo=gradio&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A generic abstractive summarization project that fine-tunes the `google/flan-t5-base` model on the SAMSum corpus using Parameter-Efficient Fine-Tuning (PEFT) and Quantized Low-Rank Adaptation (QLoRA).

## 🏗️ Project Structure

```bash
Dialogue-Summarization-T5/
├── src/                      # Source code for training and inference
│   ├── app.py                # Interactive Gradio web application (Demo)
│   ├── config.py             # Global hyperparameters and file paths
│   ├── model_loader.py       # 4-bit Quantization (NF4) & LoRA configuration
│   └── train.py              # Main training script (Hugging Face Trainer)
├── Final_Project_Fixed.ipynb # Comprehensive Project Notebook (Analysis, Training, Eval)
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```  <-- YOU MUST HAVE THESE 3 TICKS HERE TO CLOSE THE BOX!

## Here is the "Quick Start" section to the End (Copy and Paste this BELOW those 3 ticks):

```markdown
## 🚀 Quick Start

### 1. Initial Setup

First, clone the repository and enter the directory:

```bash
git clone [https://github.com/abdulmajiid1/Dialogue-Summarization-T5-.git](https://github.com/abdulmajiid1/Dialogue-Summarization-T5-.git)
cd Dialogue-Summarization-T5-













