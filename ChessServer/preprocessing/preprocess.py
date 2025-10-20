import torch
import pandas as pd
from utils import moves_to_tensors_and_info
import os
from ChessServer.cnn import ChessDualDataset
from torch.utils.data import DataLoader, random_split
import config as cf

def get_dataloader(K=1):
    path = os.path.join("data","games.csv")
    df = pd.read_csv(path)

    # FILTERS
    ELO_THRESHOLD = 1300
    MIN_GAME_LENGTH = 20
    df['num_moves'] = df['moves'].str.split().apply(len)

    filtered_df = df[(df["white_rating"] > ELO_THRESHOLD) & (df["black_rating"] > ELO_THRESHOLD)].copy()
    final_df = filtered_df[filtered_df["num_moves"] >= MIN_GAME_LENGTH]
    final_df = final_df.reset_index(drop=True) # df.iterrows went from 0 to 20K when there were only 14K relevant data

    board_tensors = []     # (7,8,8)
    info_tensors = []      # (13,1)
    labels = []            # float labels

    for idx, row in final_df.iterrows():
        # if(idx==3000):
        #     break
        moves_str = row["moves"].strip()
        if moves_str == "":
            continue

        moves_list = moves_str.split()

        for i in range(K):
            if(idx%1000==0):
                print(f"Processing for K: {i}, idx: {idx}")

            try:
                board_tensor, info_tensor = moves_to_tensors_and_info(moves_list, K=i)
            except Exception as e:
                print(f"Skipping idx={idx}, K={i} due to error: {e}")
                continue

            winner_str = row["winner"].lower()
            label = 0 if winner_str == "white" else 1

            board_tensors.append(board_tensor)
            info_tensors.append(info_tensor)
            labels.append(label)

    # Stack all tensors into batches
    X_board = torch.stack(board_tensors, dim=0).float()     # shape: (N, 7, 8, 8)
    X_info = torch.stack(info_tensors, dim=0).float()       # shape: (N, 13, 1)
    X = [(X_board[i], X_info[i]) for i in range(len(X_board))] # shape (N,2) with X[i][0] = X_board[i] and X[i][1] = X_info[i]

    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)  # shape: (N, 1)

    print(f"Total {X_board.shape[0]} examples.")
    print(f"X_board.shape = {X_board.shape}, X_info.shape = {X_info.shape}, y.shape = {y.shape}")

    # X is your list of (board_tensor, info_tensor) pairs, and y is a torch.Tensor of shape (N,1)


    full_dataset = ChessDualDataset(X, y)

    # Determine split sizes (adjust val fraction as desired; here we keep 0% val per old code)
    N = len(full_dataset)
    n_train = int(0.8 * N)
    n_test  = N - n_train
    n_val = 0
    torch.manual_seed(42)
    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=cf.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds,  batch_size=cf.BATCH_SIZE, shuffle=False, drop_last=False)

    print(f"Train size: {len(train_ds)}, Test size: {len(test_ds)}")
    return train_loader, test_loader