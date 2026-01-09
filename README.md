# 🗣️ Dialogue Summarization with Flan-T5 & QLoRA

![Python](https://img.shields.io/badge/Python-3.10-blue)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Gradio](https://img.shields.io/badge/Gradio-App-red)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview
This project focuses on **Abstractive Dialogue Summarization** using a parameter-efficient fine-tuning (PEFT) approach. We fine-tuned the `google/flan-t5-base` model on the **SAMSum corpus** using **QLoRA** (Quantized Low-Rank Adaptation). This approach allows for high-performance summarization while maintaining low computational costs.

---

## 🏗️ Project Structure
The repository is organized into a modular architecture:

```text
Dialogue-Summarization/
├── src/
│   ├── app.py                # Interactive Gradio web application (Demo)
│   ├── config.py             # Global hyperparameters and file paths
│   ├── model_loader.py       # 4-bit Quantization (NF4) & LoRA configuration
│   └── train.py              # Main training script (Hugging Face Trainer)
├── Final_Project_Fixed.ipynb # Full Project Notebook (Analysis, Training, Eval)
├── requirements.txt          # List of project dependencies
└── README.md                 # Project Documentation


🚀 Technical ImplementationModel ArchitectureBase Model: google/flan-t5-base (248M Parameters)Task: Abstractive Summarization (Seq2Seq)Optimization & Fine-TuningQuantization: BitsAndBytes 4-bit Normal Float (NF4) for memory efficiency.PEFT (LoRA): Rank=16, Alpha=32, Dropout=0.05.Target Modules: Fine-tuned q (Query) and v (Value) attention layers.DatasetSource: SAMSum Corpus (14k dialogues).Preprocessing: Tokenized with T5Tokenizer and dynamic padding.📊 Performance ResultsThe model was evaluated on the test set using standard NLP metrics. We observed no overfitting, with validation loss stabilizing at 1.40.MetricScoreNoteROUGE-147.14High overlap with reference summariesROUGE-223.50Strong bigram matchingROUGE-L39.80Good sentence structure preservationBERTScore (F1)90.95Excellent semantic similarityValidation Loss1.40Stable convergenc
e💻 How to Run1. InstallationClone the repository and install the required dependencies:Bash# Install dependencies
pip install -r requirements.txt
2. Run the Prototype (Interactive Demo)To launch the Gradio web interface and test the model summarization in real-time:Bash# Launch the App (Located in src folder)
python src/app.py
After running this command, click the local URL provided in the terminal (e.g., https://www.google.com/search?q=http://127.0.0.1:7860).3. (Optional) Retrain the ModelIf you wish to reproduce the training process from scratch:Bash# Start Training
python src/train.py
🛠️ Tools & LibrariesHugging Face Transformers: Model loading and training.PEFT (Parameter-Efficient Fine-Tuning): LoRA implementation.BitsAndBytes: 4-bit quantization.Gradio: Web interface for the prototype.Evaluate: Calculation of ROUGE and BERTScore metrics.📝 LicenseThis project is intended for educational purposes as part of the Graduate Coursework.
