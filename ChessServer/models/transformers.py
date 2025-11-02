import torch
from torch.utils.data import Dataset
from torch import nn
import torch.nn.functional as F
import chess
import pandas as pd
import config as cf
import math

class PositionalEmbedding(nn.Module):
    def __init__(self,num_tokens=64,embed_dim=1, pos_type="fixed"):
        super(PositionalEmbedding,self).__init__()
        self.pos_type = pos_type
        self.num_tokens = num_tokens
        self.embed_dim = embed_dim

        if pos_type == "learned":
            self.embedding = nn.Embedding(num_tokens, embed_dim)
        elif pos_type == "fixed":
            # 1/64, 2/64, ..., 64/64
            position_number_scaled = torch.tensor(
                [(i+1) / num_tokens for i in range(num_tokens)], 
                dtype=torch.float32
            ) #(64,)
            position_number_scaled = position_number_scaled.unsqueeze(0).unsqueeze(2) #(1,64,1)
            self.register_buffer('position_number_scaled', position_number_scaled)

        elif pos_type == "sinusoidal":
            pe = torch.zeros(num_tokens, embed_dim) #(64, 7)
            position = torch.arange(num_tokens, dtype=torch.float).unsqueeze(1) #[0, 1, 2, 3, ..., 63] and (64,1)
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            if embed_dim > 1:
                pe[:, 1::2] = torch.cos(position * div_term)[:, :embed_dim//2]
            pe = pe.unsqueeze(0) #(1,64,7)
            self.register_buffer('pe', pe)

    def forward(self, board_tensor):
        batch_size = board_tensor.size(0)
        
        if self.pos_type == "learned":
            positions = torch.arange(self.num_tokens, device=board_tensor.device).unsqueeze(0).expand(batch_size, -1) #N,64
            pos_emb = self.embedding(positions) # N,64,1
            return torch.cat([board_tensor, pos_emb], dim=2) #(N, 64, 8)
            
        elif self.pos_type == "fixed":
            pos_emb = self.position_number_scaled.expand(batch_size, -1, -1)# N,64,1
            return torch.cat([board_tensor, pos_emb], dim=2)# N,64,8
            
        elif self.pos_type == "sinusoidal":
            return board_tensor + self.pe[:, :self.num_tokens, :].to(board_tensor.device) #(N, 64, 7)
        
class InputToken(nn.Module):
    """zeroth token for metadata"""
    def __init__(self, pos_type="learned", use_info_in_zeroth=True):
        super(InputToken, self).__init__()
        self.pos_type = pos_type
        self.use_info_in_zeroth = use_info_in_zeroth

        if pos_type == "sinusoidal":
            self.pos_emb = PositionalEmbedding(num_tokens=64, embed_dim=7, pos_type=pos_type)
        else:
            self.pos_emb = PositionalEmbedding(num_tokens=64, embed_dim=1, pos_type=pos_type)
            
        if use_info_in_zeroth:
            self.info_linear = nn.Linear(13, 7)
        
    def forward(self, board_tensor, info_tensor):
        batch_size = board_tensor.size(0)
        
        board_tensor = board_tensor.permute(0, 2, 3, 1)  # Nx8x8x7
        board_tensor = board_tensor.reshape(batch_size, 64, 7)  # Nx64x7
        
        # positional embeddings
        input_token = self.pos_emb(board_tensor)  # Nx64x8
        
        if self.pos_type == "sinusoidal":
            # add one more dimension to make it 8
            padding = torch.zeros(batch_size, 64, 1, device=input_token.device)
            input_token = torch.cat([input_token, padding], dim=2)  # Nx64x8
        
        # zeroth token
        if self.use_info_in_zeroth:
            # Compress info and create zeroth token with metadata
            info_tensor = info_tensor.view(batch_size, -1).float()  # Nx13
            info_tensor = self.info_linear(info_tensor)  # Nx7
            zero_col = torch.zeros(batch_size, 1, device=info_tensor.device)
            zeroth_token = torch.cat([info_tensor, zero_col], dim=1)  # Nx8
            zeroth_token = zeroth_token.reshape(batch_size, 1, 8)  # Nx1x8
        else:
            zeroth_token = torch.zeros(batch_size, 1, 8, device=input_token.device)
        
        # Concatenate zeroth token with board tokens
        input_token = torch.cat([zeroth_token, input_token], dim=1)  # Nx65x8
        return input_token
    
class SingleHeadAttention(nn.Module):
    """self-attention """
    def __init__(self, d_model=8):
        super(SingleHeadAttention, self).__init__()
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.d_model = d_model
    
    def forward(self, input_token):
        Q = self.W_q(input_token)
        K = self.W_k(input_token)
        V = self.W_v(input_token)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_model ** 0.5)
        weights = F.softmax(attention_scores, dim=-1)
        final = torch.matmul(weights, V)
        return self.out(final)
    
class CrossAttention(nn.Module):
    """Cross-attention """
    def __init__(self, d_info=13, d_transformer=8):
        super(CrossAttention, self).__init__()
        self.W_q = nn.Linear(d_info, d_transformer)
        self.W_k = nn.Linear(d_transformer, d_transformer)
        self.W_v = nn.Linear(d_transformer, d_transformer)
        self.out = nn.Linear(d_transformer, d_transformer)
        self.d_transformer = d_transformer
        
    def forward(self, info_tensor, input_token):
        # info_tensor: (N,13), input_token: (N,65,8)
        Q = self.W_q(info_tensor.float()).unsqueeze(1).expand(-1, input_token.size(1), -1)  # N,65,8
        K = self.W_k(input_token)  # N,65,8
        V = self.W_v(input_token)  # N,65,8
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_transformer ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        final = torch.matmul(attn_weights, V)
        
        return self.out(final)
    
class TransformerBlock(nn.Module):
    """transformer block"""
    def __init__(self, attention_type="self", d_info=13, d_model=8):
        super(TransformerBlock, self).__init__()
        self.attention_type = attention_type
        
        if attention_type == "self":
            self.attention = SingleHeadAttention(d_model=d_model)
        elif attention_type == "cross":
            self.attention = CrossAttention(d_info=d_info, d_transformer=d_model)
        else:
            raise ValueError(f"Invalid attention_type: {attention_type}. Use 'self' or 'cross'")
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, d_model)
        )
    
    def forward(self, input_token, info_tensor=None):
        if self.attention_type == "self":
            attention_output = self.attention(input_token)
        elif self.attention_type == "cross":
            attention_output = self.attention(info_tensor, input_token)
        
        input_token = self.layernorm1(input_token + attention_output)
        ffn_output = self.ffn(input_token)
        output_token = self.layernorm2(input_token + ffn_output)
        
        return output_token
        
class ChessTransformerClassification(nn.Module):
    def __init__(self, pos_type="learned", attention_type="self", num_layers=4, use_info_in_zeroth=None):
        super(ChessTransformerClassification, self).__init__()
        
        self.pos_type = pos_type
        self.attention_type = attention_type
        self.num_layers = num_layers
        
        if use_info_in_zeroth is None:
            use_info_in_zeroth = (attention_type == "self")
        self.use_info_in_zeroth = use_info_in_zeroth
        
        # Input token
        self.input_tokenizer = InputToken(
            pos_type=pos_type, 
            use_info_in_zeroth=use_info_in_zeroth
        )
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(attention_type=attention_type, d_info=13, d_model=8)
            for _ in range(num_layers)
        ])
        
        # Classification head
        self.classification = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, board_tensor, info_tensor):
        input_token = self.input_tokenizer(board_tensor, info_tensor)
        
        output_token = input_token
        for transformer_block in self.transformer_blocks:
            if self.attention_type == "self":
                output_token = transformer_block(output_token)
            elif self.attention_type == "cross":
                output_token = transformer_block(output_token, info_tensor)
        
        # Extract zeroth token for classification
        zeroth_tok = output_token[:, 0, :]
        
        # Classification
        logit = self.classification(zeroth_tok)
        probability = torch.sigmoid(logit)
        
        return probability
    