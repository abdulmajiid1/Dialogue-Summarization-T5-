from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from .data_loader import load_and_process_data
from .model_loader import get_model
from .config import OUTPUT_DIR, LEARNING_RATE, BATCH_SIZE, EPOCHS

def train():
    # 1. Load Data and Model
    tokenized_dataset, tokenizer = load_and_process_data()
    model = get_model()

    # 2. Define Training Arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=EPOCHS,
        logging_steps=100,
        save_strategy="epoch",
        eval_strategy="epoch",
        predict_with_generate=True,
        report_to="tensorboard"
    )

    # 3. Initialize Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
    )

    # 4. Start Training
    print("Starting Training...")
    trainer.train()

    # 5. Save Model
    print(f"Saving model to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    train()
