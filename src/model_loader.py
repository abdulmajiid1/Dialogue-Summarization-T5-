import torch
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from .config import MODEL_ID

def get_model():
    print(f"Loading Base Model: {MODEL_ID}...")

    # 1. Define Quantization Config (4-bit)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Load Base Model
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )

    # 3. Apply LoRA (Low-Rank Adaptation)
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=16, 
        lora_alpha=32, 
        target_modules=["q", "v"], 
        lora_dropout=0.05, 
        bias="none", 
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model
