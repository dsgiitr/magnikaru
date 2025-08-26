#!apt-get install stockfish
import time
import chess
from utils import pgn_to_tensor
from model import ChessCNN
from lightning_model import LitCNN
from Transformer_cross_attention import ChessTransformerClassification
import torch


pytorch_model = ChessTransformerClassification()
model = LitCNN.load_from_checkpoint("my_model.ckpt", model=pytorch_model)

total_model_time = 0.0
total_time = 0.0
model_calls = 0


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


def predict_move(board, model, color, depth=2):
    best_move, _ = alpha_beta(board, depth, float("-inf"), float("inf"), color, model, root=True)
    return best_move

def bot_predict_move(board, color, depth=2):
   predict_best_move = predict_move(board,model,color,depth )
   return predict_best_move


STOCKFISH_PATH = "D:\\Tech\\dsg\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe"
engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
#engine.configure({"UCI_LimitStrength": True})
#engine.configure({"UCI_Elo": 1350})
engine.configure({"Skill Level": 0})
board = chess.Board()
model_color = 0

model.eval()

for i in range(1000):
  if (board.is_game_over()):
    break

  if i%2 == model_color:
    start_time = time.perf_counter()

    result = engine.play(board, chess.engine.Limit(time=0.1))
    print(f"Move: {result.move}, move type: {type(result.move)}")
    board.push(result.move)

    end_time = time.perf_counter()
    total_time += end_time - start_time
  else:
    start_time = time.perf_counter()

    my_move = predict_move(board, model, model_color)
    board.push(my_move)

    end_time = time.perf_counter()
    print(f"Time taken: {end_time - start_time:.4f} seconds")
    total_time += end_time - start_time

engine.quit()
game = chess.pgn.Game.from_board(board)
exporter = chess.pgn.StringExporter()
pgn = game.accept(exporter)
print(f"pgn:\n {pgn}")

print(f"total_time: {total_time}")
print(f"total_model_time: {total_model_time}")
print(1 - total_model_time/total_time)
print("Number of calls : ", model_calls)
print("Average model call time : ", total_model_time/model_calls)
# Even if we optimize the other portions of this code, if we don't
# decrease the number of model calls or make the model call faster,
# we can't increase depth.