```markdown
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

```

## 🚀 Quick Start

### 1. Initial Setup

First, clone the repository and enter the directory:

```bash
git clone [https://github.com/abdulmajiid1/Dialogue-Summarization-T5-.git](https://github.com/abdulmajiid1/Dialogue-Summarization-T5-.git)
cd Dialogue-Summarization-T5-

```

Next, install the required dependencies:

```bash
pip install -r requirements.txt

```

### 2. Training (Optional)

If you want to retrain the model or reproduce the fine-tuning results, run the training script:

```bash
python src/train.py

```

> **Note:** Training uses the SAMSum dataset and requires a GPU with support for BitsAndBytes quantization.

### 3. Interactive Demo

To launch the Gradio web interface for real-time summarization:

```bash
python src/app.py

```

Access the interface at:

* Local URL: [http://localhost:7860](https://www.google.com/search?q=http://localhost:7860)

## 🧪 Model & Implementation

1. **Base Model:** `google/flan-t5-base` (248M Parameters) - A powerful Seq2Seq model.
2. **Quantization:** **BitsAndBytes** 4-bit Normal Float (NF4) used for extreme memory efficiency.
3. **PEFT Strategy:** **QLoRA** (Quantized Low-Rank Adaptation) targeting the query (`q`) and value (`v`) attention layers.
4. **Dataset:** **SAMSum Corpus** (14k dialogues) processed with dynamic padding.

## 📚 Documentation

* [Analysis Notebook](https://www.google.com/search?q=Final_Project_Fixed.ipynb) - Detailed exploratory data analysis and evaluation metrics (ROUGE scores).
* [Model Config](https://www.google.com/search?q=src/config.py) - View hyperparameters, learning rates, and LoRA configurations.
* [App Source](https://www.google.com/search?q=src/app.py) - Logic behind the inference engine and UI.

## 🛠️ Tech Stack

* **Core:** Python 3.10
* **DL Framework:** PyTorch, Hugging Face Transformers
* **Optimization:** PEFT, BitsAndBytes, LoRA
* **Interface:** Gradio
* **Data Processing:** Datasets Library, Pandas

## 📝 License

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).

```

```











