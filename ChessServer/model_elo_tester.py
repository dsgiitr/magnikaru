
import time
import chess
from utils import pgn_to_tensor
from ChessServer.cnn import ChessCNN
from lightning_model import LitCNN
from transformer_learned_embedding import ChessTransformerClassification
import torch
import config as cf

MODEL_PATH = "checkpoint/kratos_checkpoint/checkpoints/epoch21_lr_0.001___2025-10-04_08-39-35.ckpt"
# Stockfish path must be verified against your system (using r-string for clean Windows path)
STOCKFISH_PATH = r"C:\Laabhanvi\DSG\magnikaru\magnikaru-monorepo\stockfish-windows-x86-64-avx2\stockfish\stockfish-windows-x86-64-avx2.exe"

pytorch_model = ChessTransformerClassification()
model = LitCNN.load_from_checkpoint(MODEL_PATH, model=pytorch_model)

elo_test_ratings = [1350, 1400, 1450]

def get_top_k_moves(board, model, k=5):
    moves = list(board.legal_moves)
    boards = []
    for move in moves:
        board.push(move)
        boards.append(board.copy())
        board.pop()

    scores = evaluate_batch(boards, model)
    move_scores = list(zip(moves, scores))
    move_scores.sort(key=lambda x: x[1], reverse=(board.turn == chess.WHITE))
    return [m for m, _ in move_scores[:k]]

def evaluate_batch(boards, model):
    games = []
    for board in boards:
        game = chess.pgn.Game.from_board(board)
        exporter = chess.pgn.StringExporter()
        pgn = game.accept(exporter)
        game_tensor, info_tensor = pgn_to_tensor(pgn, 0)
        games.append((game_tensor, info_tensor))

    game_tensors = torch.stack([g[0] for g in games])
    info_tensors = torch.stack([g[1] for g in games])

    device = next(model.parameters()).device
    game_tensors, info_tensors = game_tensors.to(device), info_tensors.to(device)

    with torch.no_grad():
        scores = model(game_tensors, info_tensors).cpu().numpy()

    return scores

def evaluate_board(board, model):
    return evaluate_batch([board], model)[0]

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


def beam_search_minimax(root_boards, depth, color, model, k=5):
    if depth == 0 or not root_boards:
        scores = evaluate_batch(root_boards, model) if root_boards else []
        return None, scores

    next_level = []
    parents = []
    for idx, board in enumerate(root_boards):
        if board.is_game_over():
            continue
        for move in get_top_k_moves(board, model, k):
            board.push(move)
            next_level.append(board.copy())
            parents.append((idx, move))
            board.pop()

    # Recursion !!!!
    _, scores = beam_search_minimax(next_level, depth-1, 1-color, model, k)

    best_scores = [float("-inf") if color==1 else float("inf")] * len(root_boards)
    best_moves = [None] * len(root_boards)

    for i in range(len(parents)):
        parent_idx, move = parents[i]
        score = scores[i]
        
        if color == 1 and score > best_scores[parent_idx]:
            best_scores[parent_idx] = score
            best_moves[parent_idx] = move
        elif color == 0 and score < best_scores[parent_idx]:
            best_scores[parent_idx] = score
            best_moves[parent_idx] = move
            
    if len(best_scores) > 0:
        final_scores = evaluate_batch(root_boards, model)
        for i in range(len(best_scores)):
            if best_moves[i] is None:
                best_scores[i] = final_scores[i]

    return best_moves, best_scores

def predict_move(board, model, color, depth=2, k=5):
    if board.is_game_over():
        return None
    
    # best_move, _ = alpha_beta(board, depth, float("-inf"), float("inf"), color, model, root=True)
    # return best_move

    moves, scores = beam_search_minimax([board], depth, color, model, k)
    
    if moves and moves[0] is not None:
        return moves[0]
    elif board.legal_moves:
        return list(board.legal_moves)[0]

# --- MODEL TESTER CLASS ---

class ModelTester:
    """chess games against Stockfish at diff Elo ratings."""
    def __init__(self, model, stockfish_path, model_depth=2):
        self.model = model
        self.stockfish_path = stockfish_path
        self.model_depth = model_depth
        self.total_games = 0
        self.total_wins = 0
        self.total_draws = 0
        self.total_losses = 0

    def run_game(self, stockfish_elo, model_is_white):
        """
        Runs a single game.
        Returns 1.0 (win), 0.5 (draw), or 0.0 (loss) from the perspective of the custom model.
        """
        board = chess.Board()
        model_color = 1 if model_is_white else 0

        engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": stockfish_elo})

        for i in range(1000):
            if (board.is_game_over()):
                break

            if i%2 == model_color:
                # start_time = time.perf_counter()

                result = engine.play(board, chess.engine.Limit(time=0.1))
                # print(f"Move: {result.move}, move type: {type(result.move)}")
                board.push(result.move)

                # end_time = time.perf_counter()
                # total_time += end_time - start_time
            else:
                # start_time = time.perf_counter()

                my_move = predict_move(board, model, model_color)
                board.push(my_move)

                # end_time = time.perf_counter()
                # print(f"Time taken: {end_time - start_time:.4f} seconds")
                # total_time += end_time - start_time

        engine.quit()

        result = board.result()
        if result == '1-0':
            return 1.0 if model_is_white else 0.0 # Model won
        elif result == '0-1':
            return 0.0 if model_is_white else 1.0 # Model lost
        else: # Draw (1/2-1/2)
            return 0.5

    def run_tournament(self, elo_ratings, games_per_elo=50):
        print("-" * 50)

        for elo in elo_ratings:
            print(f"\nTesting against Stockfish Elo: {elo}" )
            
            wins = 0
            draws = 0
            losses = 0
            
            for num in range(games_per_elo):
                model_is_white = (num % 2 == 0)
                
                score = self.run_game(elo, model_is_white)
                
                if score == 1.0:
                    wins += 1
                    status = "WIN"
                elif score == 0.5:
                    draws += 1
                    status = "DRAW"
                else:
                    losses += 1
                    status = "LOSS"
                
                print(f"  Game {num + 1:02}/{games_per_elo} (Model as {'White' if model_is_white else 'Black'}): {status}")

            total_points = wins + (0.5 * draws)
            win_rate = (wins / games_per_elo) * 100
            
            print(f"\n--- Results for Elo {elo} ---")
            print(f"total Wins: {wins}, total draws: {draws}, total losses: {losses}")
            print(f"Win Rate: {win_rate:.2f}%")

tester = ModelTester(
            model=model,
            stockfish_path=STOCKFISH_PATH,
            model_depth=2
        )
tester.run_tournament(
            elo_ratings=elo_test_ratings,
            games_per_elo=50 
        )

