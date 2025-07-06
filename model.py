import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import chess
import pandas as pd
import config as cf
import random

class ChessDualDataset(Dataset):
    def __init__(self, X, y):

        # X: list of tuples [(board_i, info_i)], where board_i is torch.Tensor (7×8×8) and info_i is torch.Tensor (13×1)
        # y: torch.Tensor of shape (N,) or (N,1) with labels 0/1

        assert len(X) == len(y)
        self.X = X
        # Ensure y is shape (N,1)
        self.y = y.view(-1, 1).float()
        
    def moves_to_tensors_and_info(moves_list, K=0):
        board = chess.Board()

        # TODO: try better error handling
        if(len(moves_list) <= K):
            return None, None # is there a better way to do this?
        trimmed_moves = moves_list[:-K] if K > 0 else moves_list


        for san in trimmed_moves:
            board.push_san(san)

        # tensor [channel, row, col] => shape (7, 8, 8)
        tensor = torch.zeros((8, 8, 8), dtype=torch.int8)

        next_to_move_flag = 0 if board.turn == chess.WHITE else 1
        tensor[0, :, :] = next_to_move_flag

        for row in range(8):
            for col in range(8):
                rank_index = 7 - row
                file_index = col
                sq = chess.square(file_index, rank_index)
                piece = board.piece_at(sq)
                if piece is not None:
                    color_flag = 0 if piece.color == chess.WHITE else 1
                    tensor[1, row, col] = color_flag

                    channel_idx = 2 + (piece.piece_type - 1)
                    tensor[channel_idx, row, col] = 1
                else:
                    tensor[1, row, col] = 0  # empty = default 0

        # Remove turn info (added to info vector)
        tensor = tensor[1:]  # shape: (7, 8, 8)

        # info_tensor: shape (13, 1)
        info_tensor = torch.zeros((13, 1), dtype=torch.int8)

        # index 0: who's turn
        info_tensor[0, 0] = next_to_move_flag

        # index 1-4: castling rights
        info_tensor[1, 0] = int(board.has_kingside_castling_rights(chess.WHITE))
        info_tensor[2, 0] = int(board.has_queenside_castling_rights(chess.WHITE))
        info_tensor[3, 0] = int(board.has_kingside_castling_rights(chess.BLACK))
        info_tensor[4, 0] = int(board.has_queenside_castling_rights(chess.BLACK))

        # index 5-12: en passant flags for files a–h (for the side to move)
        ep_square = board.ep_square
        if ep_square is not None:
            file_index = chess.square_file(ep_square)
            info_tensor[5 + file_index, 0] = 1

        return tensor, info_tensor

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        board, info = self.X[idx]
        label = self.y[idx]
        return (board, info), label

class ChessDualDatasetNew(Dataset):
    def __init__(self,train = True,K = 0):
        if train==True:
            self.csv_path = cf.TRAIN_PATH
        else:
            self.csv_path = cf.TEST_PATH
        self.K = K
        df = pd.read_csv(self.csv_path)
        self.df = df
        self.sample_probabilities = torch.ones(self.K+1)/(self.K+1)
    
    def moves_to_tensors_and_info(self,moves_list, K=0):
        board = chess.Board()

        # TODO: try better error handling
        if(len(moves_list) <= K):
            trimmed_moves = moves_list
        else:
            trimmed_moves = moves_list[:-K] if K > 0 else moves_list


        for san in trimmed_moves:
            board.push_san(san)

        # tensor [channel, row, col] => shape (7, 8, 8)
        tensor = torch.zeros((8, 8, 8), dtype=torch.int8)

        next_to_move_flag = 0 if board.turn == chess.WHITE else 1
        tensor[0, :, :] = next_to_move_flag

        for row in range(8):
            for col in range(8):
                rank_index = 7 - row
                file_index = col
                sq = chess.square(file_index, rank_index)
                piece = board.piece_at(sq)
                if piece is not None:
                    color_flag = 0 if piece.color == chess.WHITE else 1
                    tensor[1, row, col] = color_flag

                    channel_idx = 2 + (piece.piece_type - 1)
                    tensor[channel_idx, row, col] = 1
                else:
                    tensor[1, row, col] = 0  # empty = default 0

        # Remove turn info (added to info vector)
        tensor = tensor[1:]  # shape: (7, 8, 8)

        # info_tensor: shape (13, 1)
        info_tensor = torch.zeros((13, 1), dtype=torch.int8)

        # index 0: who's turn
        info_tensor[0, 0] = next_to_move_flag

        # index 1-4: castling rights
        info_tensor[1, 0] = int(board.has_kingside_castling_rights(chess.WHITE))
        info_tensor[2, 0] = int(board.has_queenside_castling_rights(chess.WHITE))
        info_tensor[3, 0] = int(board.has_kingside_castling_rights(chess.BLACK))
        info_tensor[4, 0] = int(board.has_queenside_castling_rights(chess.BLACK))

        # index 5-12: en passant flags for files a–h (for the side to move)
        ep_square = board.ep_square
        if ep_square is not None:
            file_index = chess.square_file(ep_square)
            info_tensor[5 + file_index, 0] = 1

        return tensor, info_tensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        moves_str = row['moves'].strip()
        if not moves_str:
            raise IndexError(f"Empty moves string at index {idx}")

        sample_K = random.choices(range(self.K+1), weights=self.sample_probabilities)[0]
        # print(sample_K,end=" ")
        moves_list = moves_str.split()
        board_tensor, info_tensor = self.moves_to_tensors_and_info(moves_list, K=sample_K)
        if board_tensor is None or info_tensor is None:
            raise IndexError(f"Could not parse moves at index {idx} with K={sample_K}")

        # Label: 0 for white win, 1 for black win
        winner = row['winner'].lower()
        label = 0 if winner == 'white' else 1
        label_tensor = torch.tensor([label], dtype=torch.float32)

        return (board_tensor.float(), info_tensor.float()), label_tensor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.2):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_p)
        )
        # 1×1 projection if channels differ
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        out = self.conv(x)
        skip = self.shortcut(x)
        return F.relu(out + skip)

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.block1 = ResidualBlock(7, 32, dropout_p=0.2)
        self.block2 = ResidualBlock(32, 64, dropout_p=0.2)
        self.block3 = ResidualBlock(64, 128, dropout_p=0.2)

        # Info NN
        self.info_fc1 = nn.Linear(13, 128)
        self.info_fc2 = nn.Linear(128, 64)

        # FC
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.dropout_fc1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, board_tensor, info_tensor):

        # board_tensor: (batch, 7, 8, 8)
        # info_tensor:  (batch, 13, 1)

        x = self.block1(board_tensor)    # (batch, 32, 8, 8)

        # Info NN
        info = info_tensor.view(info_tensor.size(0), -1)      # (batch,13)
        info = F.relu(self.info_fc1(info))                    # (batch,128)
        info = F.relu(self.info_fc2(info))                    # (batch,64)
        info_bias = info.view(-1, 1, 8, 8)                    # (batch,1,8,8)

        x = x + info_bias.expand(-1, x.size(1), -1, -1)       # (batch,32,8,8)

        # Continue CNN
        x = self.block2(x)                                      # (batch, 64, 8, 8)
        x = self.block3(x)                                      # (batch,128, 8, 8)

        # FC
        x = x.view(x.size(0), -1)                               # (batch, 128*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc1(x)
        x = torch.sigmoid(self.fc2(x))                         # (batch,1)

        return x