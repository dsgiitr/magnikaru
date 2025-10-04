import time
import chess
from utils import pgn_to_tensor
from model import ChessCNN
from lightning_model import LitCNN
from Transformer_cross_attention import ChessTransformerClassification
import torch


pytorch_model = ChessCNN()
# model = LitCNN.load_from_checkpoint("amx_transformer_cross_attention_epoch_10_lr_0.001_2025-08-27_09-06-38.ckpt", model=pytorch_model)
model = LitCNN.load_from_checkpoint("Laabhanvi_CNN_epoch_10_lr_0.001_2025-08-28_16-31-16.ckpt", model=pytorch_model)

total_model_time = 0.0
total_time = 0.0
model_calls = 0

def alpha_beta(board, depth, alpha, beta, color, model, root=True):
    # Returns best_move, prob of white winning
    if depth == 0 or board.is_game_over():
        p_white = evaluate_board(board, model)
        return (None, p_white)

    best_move = None
    if color == 1:
        value = float("-inf")
        for move in board.legal_moves:
            board.push(move)
            _, eval_score = alpha_beta(board, depth - 1, alpha, beta, 0, model, root=False)
            board.pop()
            if eval_score > value:
                value = eval_score
                best_move = move
            alpha = max(alpha, value)
            if beta <= alpha:
                break
    else:
        value = float("inf")
        for move in board.legal_moves:
            board.push(move)
            _, eval_score = alpha_beta(board, depth - 1, alpha, beta, 1, model, root=False)
            board.pop()
            if eval_score < value:
                value = eval_score
                best_move = move
            beta = min(beta, value)
            if beta <= alpha:
                break

    return (best_move, value) if root else (None, value)


# BOT SAN MOVE PREDICTION
def evaluate_board(board, model):
  global total_model_time # TO REMOVE
  global model_calls

  game = chess.pgn.Game.from_board(board)
  exporter = chess.pgn.StringExporter()
  pgn = game.accept(exporter)

  game_tensor, info_tensor = pgn_to_tensor(pgn, 0)
  game_tensor = game_tensor.unsqueeze(0)
  info_tensor = info_tensor.unsqueeze(0)

  with torch.no_grad():
    start_time = time.perf_counter() # TO REMOVE
    score = model(game_tensor, info_tensor).item()
    end_time = time.perf_counter() # TO REMOVE
    total_model_time += end_time - start_time # TO REMOVE
    model_calls += 1 # TO REMOVE

  return score

def bot_evaluate_board(board):
    return evaluate_board(board, model)


# BOT SAN MOVE PREDICTION
def predict_move(board, model, color, depth=2):
    best_move, _ = alpha_beta(board, depth, float("-inf"), float("inf"), color, model, root=True)
    return best_move

def bot_predict_move(board, color, depth=2):
   predict_best_move = predict_move(board,model,color,depth )
   return predict_best_move

