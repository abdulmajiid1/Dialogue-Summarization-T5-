# Global Configuration
MODEL_ID = "google/flan-t5-base"
DATASET_ID = "knkarthick/samsum"
OUTPUT_DIR = "flan-t5-samsum-lora"

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 1
LEARNING_RATE = 1e-3
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 128
