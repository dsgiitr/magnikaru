import os

NUM_EPOCHS   = 15
BATCH_SIZE   = 128 #1024
LEARNING_RATE = 1e-3
NUM_WORKERS = 16
RANDOM_STATE = 42
SEED = 42

MIN_STEP = 10

DATA_FILE = os.path.join("data","games.csv")
TRAIN_PATH = os.path.join("data/GM_training_dataset.csv")
TEST_PATH  = os.path.join("data/GM_validation_dataset.csv")
MODEL_NAME="transformer_larned_embedding.py"

CHECKPOINT_PATH= "checkpoints/epoch0_lr_0.001___2025-10-29_18-28-20.ckpt"
