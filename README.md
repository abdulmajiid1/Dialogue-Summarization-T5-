# Dialogue Summarization with Flan-T5 & QLoRA

## 📌 Project Overview
This project focuses on **Abstractive Dialogue Summarization** using a parameter-efficient fine-tuning (PEFT) approach. We fine-tuned the `google/flan-t5-base` model on the **SAMSum corpus** using **QLoRA** (Quantized Low-Rank Adaptation) to achieve high performance with limited computational resources.



## 📂 Repository Structure
This project follows a professional modular architecture:

```text
Dialogue-Summarization/
├── app/
│   └── app.py                # Interactive Gradio web application (Demo)
├── src/
│   ├── config.py             # Global hyperparameters and file paths
│   ├── model_loader.py       # 4-bit Quantization (NF4) & LoRA configuration
│   └── train.py              # Main training script (Hugging Face Trainer)
├── Final_Project_Fixed.ipynb # Full Project Notebook (Data Analysis, Training, Evaluation)
├── requirements.txt          # List of dependencies
└── README.md                 # Project Documentation

🚀 Technical Implementation
Base Model: Flan-T5 Base (248M Parameters)

Optimization:

BitsAndBytes: 4-bit Normal Float (NF4) quantization.

LoRA: Rank=16, Alpha=32, targeting Query/Value modules.

Dataset: SAMSum (14k training dialogues).

📊 Results
The model was evaluated on the test set with the following metrics:

ROUGE-1: 47.14

BERTScore (F1): 90.95

Validation Loss: 1.40 (No overfitting observed)

💻 How to Run
1. Installation
Clone the repository and install the required dependencies:

Bash

pip install -r requirements.txt
2. Run the Prototype (Interactive Demo)
To launch the Gradio web interface and test the model:

Bash

python app/app.py
Click the local URL provided in the terminal (e.g., https://www.google.com/search?q=http://127.0.0.1:7860) to open the app.

3. (Optional) Retrain the Model
To reproduce the training process from scratch:

Bash

python src/train.py


