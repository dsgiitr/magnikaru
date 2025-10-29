import os

NUM_EPOCHS   = 15
BATCH_SIZE   = 128
LEARNING_RATE = 1e-3
NUM_WORKERS = 16
RANDOM_STATE = 42
SEED = 42

MIN_STEP = 10

DATA_FILE = os.path.join("data","games.csv")
TRAIN_PATH = os.path.join("data/GM_training_dataset")
TEST_PATH  = os.path.join("data/GM_validation_dataset")

# CHECKPOINT_PATH= "lightning_logs/version_0/checkpoints/epoch=2-step=15984.ckpt"
