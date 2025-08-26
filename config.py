import os

NUM_EPOCHS   = 2
BATCH_SIZE   = 128
LEARNING_RATE = 1e-3
NUM_WORKERS = 8
RANDOM_STATE = 42
SEED = 42

MIN_STEP = 10

DATA_FILE = os.path.join("data","games.csv")
SAVE_FILE = os.path.join("lightning_check","trainNew.ckpt")
TRAIN_PATH = os.path.join("data/train_asutosh.csv")
TEST_PATH  = os.path.join("data/test_asutosh.csv")


TEST_K = 10