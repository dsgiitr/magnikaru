# Magnikaru 
---
# Setup instructions
## Chess Server
```
> cd ChessServer
```
### Create Environment
```
> conda create -n env_name python=3.12
> conda activate env_name
```

### Install dependancies
```
> pip install -r req-torch.txt
```
If the Above doesn't work then install the packages manually

```
> pip install numpy matplotlib pandas scikit-learn tqdm chess
> flask flask_cors
```

Pytorch installation (Changes from time to time) so refer the docs
```
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
> pip install lightning
```
### Start the server 

Make sure you have installed the dependancies correctly

Start the server on port 8900 (this is a must!)
```
> python app.py
```


### NOTE !!!:
1. Make sure that you have the right version of Cuda installed inside the environment. Use use the following commands to check for required informations!
```
> nvidia-smi 
```
## Magnikaru app
Simple Chess Server using flask, chessboard2.js and chess.js

### Setup Instructions
You 2 have choices

### 1. Docker
```
> docker compose up --build
```

### 2. Manual Installation
### Environment
```
> virtualenv venv
> .\venv\Scripts\activate
```

### Install Dependancies
```
> pip install -r requirements.txt
```

### Run the Server
```
> flask run
```

<hr/>
Frontend made with ❣️by AMX






