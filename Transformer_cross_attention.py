#  meta data --> cross attention
#  positional embedding --> concatenate
# Transformer_cross_attention
import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import chess
import pandas as pd
import config as cf
import random

# CNN
"""class ResidualBlock(nn.Module):
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

        return x"""


class PositionalEmbedding(nn.Module):
    def __init__(self,num_tokens=64):
        super(PositionalEmbedding,self).__init__()
        self.position_number_scaled= torch.tensor([(i+1 )/ (num_tokens ) for i in range(num_tokens)],dtype=torch.float32)
        self.position_number_scaled =  self.position_number_scaled.unsqueeze(0).unsqueeze(2) # shape → (1, 64, 1)
        # CPU GPU MISMATCHHHHHH!!! 
        
    def forward(self,board_tensor):
        batch_size = board_tensor.size(0)
        pos_emb = self.position_number_scaled.to(board_tensor.device)  # match device
        pos_emb = pos_emb.expand(batch_size, -1, -1)
        return torch.cat([board_tensor, pos_emb], dim=2)
    

class InputToken(nn.Module):
    def __init__(self):
        super(InputToken,self).__init__()
        self.pos_emb = PositionalEmbedding(num_tokens=64)
        self.info_linear=nn.Linear(13,7)
        

    def forward(self,board_tensor,info_tensor):
        
        batch_size = board_tensor.size(0)
        board_tensor=board_tensor.permute(0,2,3,1) # Nx8x8x7
        board_tensor=board_tensor.reshape(batch_size,64,7) #Nx64x7 

        input_token=self.pos_emb(board_tensor) # Nx64x8
        zero_token = torch.zeros(batch_size, 1, input_token.size(-1), device=input_token.device)
        input_token = torch.cat([zero_token, input_token], dim=1)  # (N,65,8)

        return input_token
    
class CrossAttention(nn.Module):
    def __init__(self, d_info=13, d_transformer=8):
        super(CrossAttention, self).__init__()
        self.W_q = nn.Linear(d_info, d_transformer)
        self.W_k = nn.Linear(d_transformer, d_transformer)
        self.W_v = nn.Linear(d_transformer, d_transformer)
        self.out = nn.Linear(d_transformer, d_transformer)
        self.layernorm1 = nn.LayerNorm(d_transformer)
        self.ffn = nn.Sequential(
            nn.Linear(d_transformer, 32),
            nn.ReLU(),
            nn.Linear(32, d_transformer)
        )

    def forward(self, info_tensor, input_token):
        # info_tensor: (N,13)
        # board_tokens: (N,65,8)

        Q = self.W_q(info_tensor.float()).unsqueeze(1).expand(-1, input_token.size(1), -1)  # (N,65,8)
        K = self.W_k(input_token)              # N,65,8
        V = self.W_v(input_token)              # N,65,8

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / 8**0.5  # N,65,65
        attn_weights = F.softmax(attn_scores, dim=-1)  #N,65,65

        final = torch.matmul(attn_weights, V)  #N,65,8
        

        x = self.layernorm1(Q + self.out(final))
        ffn_out = self.ffn(x)

        out = self.layernorm1(x + ffn_out) #N,65,8
        return out
    
class ChessTransformerClassification(nn.Module):
    def __init__(self):
        super(ChessTransformerClassification, self).__init__()
        self.input_tokenizer = InputToken()
        self.cross_block = CrossAttention(d_info=13, d_transformer=8)
        self.classification = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, board_tensor, info_tensor):
        board_tokens = self.input_tokenizer(board_tensor, info_tensor)  

        cross_out = self.cross_block(info_tensor, board_tokens) 
        cls_token = cross_out[:, 0, :] #N,8

        logit = self.classification(cls_token)
        probability = torch.sigmoid(logit)
        return probability
