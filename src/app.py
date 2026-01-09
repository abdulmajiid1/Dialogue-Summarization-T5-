import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel
import torch

# NOTE: When you run this script, it will generate a TEMPORARY public link.
# You must run this script ON THE DAY of your presentation.

# 1. Configuration
# In a real setup, we point to the saved model folder.
# For this demo to work in Colab, we assume the model is in Google Drive or local.
MODEL_PATH = "../flan-t5-samsum-lora" 

print("⏳ Loading the model... (This takes about 1 minute)")

try:
    # Load Base Model & Tokenizer
    base_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", device_map="auto")
    model = PeftModel.from_pretrained(base_model, MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Ensure you have trained the model and the path is correct.")

# 2. Summarization Logic
def summarize_dialogue(dialogue):
    if not dialogue:
        return "Please enter a dialogue."
        
    prompt = "Summarize the following conversation:\n" + dialogue
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, 
            max_new_tokens=100, 
            num_beams=5
        )
        
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 3. The Web Interface
demo = gr.Interface(
    fn=summarize_dialogue,
    inputs=gr.Textbox(lines=10, label="Paste Dialogue Here", placeholder="A: Hi...\nB: Hello..."),
    outputs=gr.Textbox(label="Summary"),
    title="Dialogue Summarization System",
    description="Final Project Demo: Fine-Tuned Flan-T5 with LoRA Adapters."
)

if __name__ == "__main__":
    # share=True generates the public link for your presentation
    demo.launch(share=True)
