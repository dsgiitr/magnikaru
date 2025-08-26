from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from engine import bot_predict_move, bot_evaluate_board
import chess
import random
import json

app = Flask(__name__)
CORS(app)

def bot_move(current_fen):
    board = chess.Board(current_fen)
    print(board)
    COLOR = 1 # 1 for black and 0 for white
    # TODO: Change this to get the current color from the FEN String
    bot_score = bot_evaluate_board(board)
    print(f"Bot score: {bot_score}")
    bot_prediction_uci = bot_predict_move(board, COLOR,depth=3)
    print(f"bot_predict_move: {bot_prediction_uci}, type: {type(bot_prediction_uci)}")
    print(f"SAN: {board.san(bot_prediction_uci)}")
    
    san_move = board.san(bot_prediction_uci)
    score = bot_score*(1600) - 800
    return san_move, score

@app.route('/api/make_move', methods = ["POST","GET"]) 
def make_move():
    data_json_string = request.json # json string
    data = json.loads(data_json_string)
    
    move, score = bot_move(data["fen"])
    return jsonify({
        "move":move,
        "score":score,
        "message":f"Random move from server for input fen: {data['fen']}"
    })

if(__name__=="__main__"): 
    PORT = 8900
    print(f"Running server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)