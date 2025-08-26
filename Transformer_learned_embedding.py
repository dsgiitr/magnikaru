#Transformer_learned_embedding
import torch
from torch import nn
import torch.nn.functional as F

import config as cf
import random

class PositionalEmbedding(nn.Module):
    def __init__(self,num_tokens=64,embed_dim=1):
        super(PositionalEmbedding,self).__init__()
        # Learnable embedding
        self.embedding = nn.Embedding(num_tokens, embed_dim)
        
    def forward(self,board_tensor):
        batch_size = board_tensor.size(0)
        positions = torch.arange(64, device=board_tensor.device).unsqueeze(0).expand(batch_size, -1) #N,64

        
        pos_emb = self.embedding(positions)   

        return torch.cat([board_tensor, pos_emb], dim=2)  
    

class InputToken(nn.Module):
    def __init__(self):
        super(InputToken,self).__init__()
        self.pos_emb = PositionalEmbedding(num_tokens=64, embed_dim=1)
        self.info_linear=nn.Linear(13,7)
        

    def forward(self,board_tensor,info_tensor):
        
        batch_size = board_tensor.size(0)
        board_tensor=board_tensor.permute(0,2,3,1) # Nx8x8x7
        board_tensor=board_tensor.reshape(batch_size,64,7) #Nx64x7 

        
        input_token=self.pos_emb(board_tensor) # Nx64x8

        info_tensor = info_tensor.view(batch_size, -1) #Nx13
        info_tensor=self.info_linear(info_tensor) # Nx7

        zero_col = torch.zeros(batch_size, 1, device=info_tensor.device) 
        zeroth_token=torch.cat([info_tensor,zero_col],dim=1) # Nx8
        zeroth_token=zeroth_token.reshape(batch_size,1,8) # Nx1x8

        # Now adding zeroth token at front

        input_token=torch.cat([zeroth_token,input_token],dim=1) #Nx65x8
        return input_token

class SingleHeadAttention(nn.Module):
    def __init__(self):
        super(SingleHeadAttention,self).__init__()
        self.W_q = nn.Linear(8, 8) 
        self.W_k = nn.Linear(8, 8)
        self.W_v = nn.Linear(8, 8)
        self.out = nn.Linear(8, 8)
    
    def forward (self,input_token):
        Q= self.W_q(input_token)
        K= self.W_k(input_token)          
        V= self.W_v(input_token)

        attention_scores=torch.matmul(Q,K.transpose(-2,-1)) / 8**0.5 # Q.K^T / root(64)   # Nx65x65
        weights=F.softmax(attention_scores,dim=-1)
        final=torch.matmul(weights,V) # Nx65x8
        return self.out(final)

class TransformerBlock(nn.Module):
    def __init__(self):
        super(TransformerBlock,self).__init__()
        self.attention=SingleHeadAttention()
        self.layernorm=nn.LayerNorm(8)

        self.ffn=nn.Sequential(nn.Linear(8,32),
                               nn.ReLU(),
                               nn.Linear(32,8))


    def forward(self,input_token):
        attention_output=self.attention(input_token)
        input_token=input_token+attention_output
        input_token=self.layernorm(input_token)

        ffn=self.ffn(input_token)
        output_token=self.layernorm(input_token+ffn)

        return output_token
    

class ChessTransformerClassification(nn.Module):
    def __init__(self):
        super(ChessTransformerClassification,self).__init__()
        self.input_tokenizer = InputToken()
        self.transformer=TransformerBlock()
        self.classification=nn.Sequential(nn.Linear(8,32),
                                          nn.ReLU(),
                                          nn.Linear(32,1))

    def forward(self,board_tensor,info_tensor):
        input_token = self.input_tokenizer(board_tensor, info_tensor)

        output_transformer=self.transformer(input_token)
        zeroth_tok=output_transformer[:,0,:] 
        logit=self.classification(zeroth_tok)
        probability=torch.sigmoid(logit)
        return probability
