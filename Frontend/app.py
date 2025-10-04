from flask import Flask, render_template, jsonify, request
import requests
from flask_cors import CORS
import chess
import random
import json
from dotenv import load_dotenv
import os
load_dotenv()
app = Flask(__name__)
CORS(app)

# CHESS FUNCTIONS
def make_random_move(current_fen):
    board = chess.Board(current_fen)
    print(board)
    leg_moves = list(board.legal_moves)
    print(f"Moves: {leg_moves}")
    if len(leg_moves)==0:
        return ""
    move_idx = random.randint(0,len(leg_moves)-1)
    # move_idx = 0
    uci_move = leg_moves[move_idx]
    
    san_move = board.san(uci_move)
    score = random.random()*(800)
    return san_move, score


# ROUTES
@app.route('/') 
@app.route('/botplay') 
def botplay():
    return render_template("PlayerBot.html")

@app.route('/api/random_move', methods=['POST']) 
def random_move():
    '''
    {
        "pgn":"",
        "fen":""
    }
    '''
    data = request.json
    move,score = make_random_move(data['fen'])
    print(f"Random Move from server: {move}")
    return jsonify({
        "move":move,
        "score":score,
        "message":f"Random move from server for input fen: {data['fen']}"
    })

@app.route('/api/get_server_move', methods=['POST']) 
def get_server_move():
    """
    Makes request to model server to get the best move and eval score
    """
    data = request.json
    print(f"JSON from frontend: {data}")
    api_url = "http://localhost:8900/api/make_move"  # request to the Model Server
    payload = {
        "fen": data["fen"]
    }
    print(f"Request to CHESS SERVER {payload} ")
    try:
        response = requests.post(api_url, json=json.dumps(payload))
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()
        return jsonify(data)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500


if(__name__=="__main__"): 
    PORT = os.getenv("PORT")
    print(f"Running server on port {PORT}")
    app.run(host='0.0.0.0', port=PORT)