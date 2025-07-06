import os
import pandas as pd
from sklearn.model_selection import train_test_split
import config as cf

DATA_PATH = cf.DATA_FILE
TRAIN_PATH = cf.TRAIN_PATH
TEST_PATH = cf.TEST_PATH
ELO_THRESHOLD = 1300
MIN_GAME_LENGTH = 20
CHUNK_SIZE = cf.BATCH_SIZE  
RANDOM_STATE = cf.RANDOM_STATE
TEST_SIZE = 0.2  # proportion for train_test_split

# Avoid Duplicates
for path in (TRAIN_PATH, TEST_PATH):
    if os.path.exists(path):
        os.remove(path)

# Process file in chunks
reader = pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE)
first_train_chunk = True
first_test_chunk = True

for chunk in reader:
    # Filtering
    chunk['num_moves'] = chunk['moves'].str.split().apply(len)
    filtered = chunk[
        (chunk['white_rating'] > ELO_THRESHOLD) &
        (chunk['black_rating'] > ELO_THRESHOLD) &
        (chunk['num_moves'] >= MIN_GAME_LENGTH)
    ].reset_index(drop=True)

    if filtered.empty:
        continue

    # Split into train and test
    train_chunk, test_chunk = train_test_split(
        filtered,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Append to train.csv
    train_chunk.to_csv(
        TRAIN_PATH,
        mode='a',
        index=False,
        header=first_train_chunk
    )
    first_train_chunk = False

    # Append to test.csv
    test_chunk.to_csv(
        TEST_PATH,
        mode='a',
        index=False,
        header=first_test_chunk
    )
    first_test_chunk = False

print(f"Data processing complete. Outputs:\n- {TRAIN_PATH}\n- {TEST_PATH}")

full_train = pd.read_csv(TRAIN_PATH)
full_test = pd.read_csv(TEST_PATH)
print(f"Train Info. Outputs:\n {full_train.info()}")
print(f"Test Info. Outputs:\n {full_test.info()}")
