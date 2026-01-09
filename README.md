# Dialogue Summarization with Flan-T5 & QLoRA

## 📌 Project Overview
This project focuses on **Abstractive Dialogue Summarization** using a parameter-efficient fine-tuning (PEFT) approach. We fine-tuned the `google/flan-t5-base` model on the **SAMSum corpus** using **QLoRA** (Quantized Low-Rank Adaptation) to achieve high performance with limited computational resources.

## 📂 Repository Structure
This project follows a professional modular architecture:

| File / Folder | Description |
|:--- |:--- |
| **`src/`** | Source code for training and data processing. |
| **`src/config.py`** | Global hyperparameters and file paths. |
| **`src/model_loader.py`** | Implementation of 4-bit Quantization (NF4) and LoRA configuration. |
| **`src/train.py`** | Main training script using Hugging Face Trainer. |
| **`app/app.py`** | Interactive Gradio web application for demonstration. |
| **`requirements.txt`** | List of project dependencies. |

## 🚀 Technical Implementation
* **Base Model:** Flan-T5 Base (248M Parameters)
* **Optimization:**
    * **BitsAndBytes:** 4-bit Normal Float (NF4) quantization.
    * **LoRA:** Rank=16, Alpha=32, targeting Query/Value modules.
* **Dataset:** SAMSum (14k training dialogues).

## 📊 Results
The model was evaluated on the test set with the following metrics:
* **ROUGE-1:** 47.14
* **BERTScore (F1):** 90.95
* **Validation Loss:** 1.40 (No overfitting observed)

## 💻 How to Run

### 1. Installation
Clone the repository and install the required dependencies:
```bash
pip install -r requirements.txt
