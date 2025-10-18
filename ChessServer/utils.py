import chess
import chess.pgn
import torch

from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
import pandas as pd
import random
import re
import io
import config as cf

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

class ChessDataset(IterableDataset):
    def __init__(self, end_steps:int, train_csv:str, test_csv:str,sampling_probabilities = None,scale=2.0, mode='test'):
        #  end_step = K
        self.end_steps = end_steps
        self.scale=scale
        self.csv_path = train_csv if mode == 'train' else test_csv
        self.chunksize = 10_000
        # self.sampling_probabilities = sampling_probabilities if sampling_probabilities is not None else torch.ones(end_steps+1)/(end_steps+1)
        if sampling_probabilities is not None:
            self.sampling_probabilities = sampling_probabilities
        else:
           
            weights = torch.exp(torch.linspace(0, scale, end_steps+1)) #exponential probabilities
            self.sampling_probabilities = weights / weights.sum()

        self.mode = mode  # 'train' or 'test'

    def __iter__(self):
        worker_info = get_worker_info()
        #print(worker_info)
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        for chunk in pd.read_csv(self.csv_path, chunksize=self.chunksize):
            for idx, (pgn, result) in enumerate(zip(chunk["pgn"], chunk["Result"])):

                if idx % num_workers != worker_id:
                    continue
                label = int(result[0])
                label_tensor = torch.tensor([label], dtype=torch.float32)
                if self.mode == 'train':
                    weights = self.sampling_probabilities.tolist()
                    chosen_end_step = random.choices(range(self.end_steps+1), weights=weights)[0]
                    # Potential for inconsistency here, chosen_end_step can be greater than game
                    # size which will be clipped but won't be logged properly... fix later
                    game, info = pgn_to_tensor(pgn, chosen_end_step)
                    # yield (game.float(), info.float()), label_tensor , chosen_end_step
                    yield -1, (game.float(), info.float()), label_tensor
                elif self.mode == 'test':
                    for i in range(self.end_steps+1): # Can become problematic here if end_step exceeds size of game
                        # i = self.end_steps
                        game, info = pgn_to_tensor(pgn, i)
                        # print(f"K: {i}")
                        # i is the Value of K
                        yield i, (game.float(), info.float()), label_tensor
                else:
                    print("Invalid Mode provided!")

def pgn_to_tensor(pgn, end_steps):
    game = chess.pgn.read_game(io.StringIO(pgn))
    board = game.board()

    total_moves = sum(1 for _ in game.mainline_moves())
    
    min_step = cf.MIN_STEP
    moves_required = max(min_step, total_moves - end_steps)

    count = 0
    for move in game.mainline_moves():
        if count == moves_required:
            break
        board.push(move)
        count+=1

    fen = board.fen()

    info_tensor = torch.zeros(13, dtype=torch.int8)

    # index 0: who's turn
    info_tensor[0] = 1 if (board.turn == chess.WHITE) else 0

    # index 1-4: castling rights
    info_tensor[1] = int(board.has_kingside_castling_rights(chess.WHITE))
    info_tensor[2] = int(board.has_queenside_castling_rights(chess.WHITE))
    info_tensor[3] = int(board.has_kingside_castling_rights(chess.BLACK))
    info_tensor[4] = int(board.has_queenside_castling_rights(chess.BLACK))

    # index 5-12: en passant flags for files a–h (for the side to move)
    ep_square = board.ep_square
    if ep_square is not None:
        file_index = chess.square_file(ep_square)
        info_tensor[5 + file_index] = 1

    return fen_to_tensor(fen), info_tensor

# [CurrentPosition "r5k1/3R4/p1p3pB/5p2/2P4P/6P1/PP6/6K1 w - -"] // Be very very very sure that this format doesn't change

def fen_to_tensor(fen):

    channels = {'b': 1, 'k':2, 'n':3, 'p':4, 'q':5, 'r':6}

    game = torch.zeros((7,8,8))
    rows = re.split(r'[/\s]+', fen)
    #assert(len(rows) == 11) Letting this condition go for now

    for row_index in range(8):
        row = rows[row_index]
        col_index = 0
        for c in row:
            if c.isdigit():
                for _ in range(int(c)):
                    game[0][row_index][col_index] = 0.5
                    col_index += 1
            else:
                game[0][row_index][col_index] = 1 if c.isupper() else 0
                game[channels[c.lower()]][row_index][col_index] = 1
                col_index += 1

    return game # Returns Channels*height*width