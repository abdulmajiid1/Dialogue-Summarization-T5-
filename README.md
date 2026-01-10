# Dialogue-Summarization-T5-

## 📌 Project Overview

This project focuses on **Abstractive Dialogue Summarization** using a parameter-efficient fine-tuning (PEFT) approach. We fine-tuned the `google/flan-t5-base` model on the **SAMSum corpus** using **QLoRA** (Quantized Low-Rank Adaptation). This approach allows for high-performance summarization while maintaining low computational costs.

## 🏗️ Project Structure

```bash

```text
Dialogue-Summarization-T5/
│
├── Notebook/                               # Jupyter Notebooks Directory
│   └── Final_Project_Fixed (2).ipynb       # Main Project Notebook
│
├── src/                                    # Source Code Directory
│   ├── app.py                             # Interactive Gradio App
│   ├── train.py                           # Training Script
│   └── config.py                          # Configuration Settings
│
├── requirements.txt                       <-- List of Dependencies
├── README.md                              <-- Project Documentation
└── LICENSE                                <-- MIT License

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

This project is for educational purposes. If you use external datasets, please check their original license terms



