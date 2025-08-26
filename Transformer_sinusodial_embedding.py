#Transformer_sinusodial_embedding
import torch
import math

# Positional Encodings
class ChessPositionalEncoding(torch.nn.Module):
  def __init__(self, depth: int, max_tokens=64,mode="periodic"):
    super(ChessPositionalEncoding, self).__init__()
    '''
    depth: depth of each chess cell
    mode: periodic | fixed (concat the on dth index)
    '''
    self.depth = depth
    self.mode = mode
    self.max_tokens = max_tokens 

    if self.mode == "periodic":
      # Add the sine and cosine positional embeddings
      pe = torch.zeros(self.max_tokens, depth)
      position = torch.arange(self.max_tokens, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, depth, 2).float() * (-math.log(10000.0) / depth))
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.cos(position * div_term)[:,:depth//2]
      pe.unsqueeze_(0)
      print(f"Shape of positional Encoding: {pe.shape}")
    elif self.mode == "fixed":
      # Concatenate position index to dth index (0 to d-1 index for depth)
      pe = torch.zeros(self.max_tokens, 1)
      position = torch.arange(0, self.max_tokens, dtype=torch.float)
      pe = position.reshape(self.max_tokens,-1)
      pe.unsqueeze_(0)
      print(f"Shape of positional Encoding: {pe.shape}")
    else:
      assert False, f"Expected 'periodic' or 'fixed', got invalid mode {self.mode}!"
    
    self.register_buffer('pe', pe)    

  def forward(self, x):
    '''
    x: (batch_size, rc, depth)
    '''
    if self.mode == "periodic":
      x = x + self.pe[:, :]
    elif self.mode == "fixed":
      x = torch.cat((x, self.pe.repeat(x.shape[0],1,1)), dim=2)
    else:
      assert False, f"Expected 'periodic' or 'fixed', got invalid mode {self.mode}!"

    return x
# Multi Head Attention
class MHA(torch.nn.Module):
  def __init__(self, depth: int, n_heads=1):
    '''
    depth: depth of each chess cell
    n_heads: number of heads
    '''
    super(MHA, self).__init__()
    assert depth % n_heads == 0, "depth must be divisible by n_heads"

    self.depth = depth
    self.n_heads = n_heads
    self.d_k = depth // n_heads

    self.W_q = torch.nn.Linear(depth, depth)
    self.W_k = torch.nn.Linear(depth, depth)
    self.W_v = torch.nn.Linear(depth, depth)

    self.W_o = torch.nn.Linear(depth, depth) # weights for the concatenated Heads

  def split_heads(self, x):
    '''
    x: (batch_size, rc, depth) [Q, K, V]

    return: (batch_size, n_heads, rc, d_k)
    '''
    batch_size, rc, depth = x.size()
    return x.view(batch_size, rc, self.n_heads, self.d_k).transpose(1, 2)

  def combine_heads(self, x):
    '''
    x: (batch_size, n_heads, rc, d_k)

    return: (batch_size, rc, depth)
    '''
    batch_size, n_heads, rc, d_k = x.size()
    return x.transpose(1, 2).contiguous().view(batch_size, rc, self.depth) # self.depth = n_heads * d_k

  def attention(self, Q, K, V):
    '''
    Q: (batch_size, n_heads, rc, d_k)
    K: (batch_size, n_heads, rc, d_k)
    V: (batch_size, n_heads, rc, d_k)

    attention_heads: (batch_size, n_heads, rc, d_k)
    attention_probabilities: (batch_size, n_heads, rc, rc)

    return attention_heads, attention_probabilities
    '''
    attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
    # TODO: Apply Mask

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
    return torch.matmul(attention_probs, V), attention_probs

  def forward(self, Q, K, V, mask=None):
    '''
    Q: (batch_size, rc, depth)
    K: (batch_size, rc, depth)
    V: (batch_size, rc, depth)
    mask: (batch_size, rc, rc) # Not used for now
    '''
    # Split into Heads
    Q = self.split_heads(Q)
    K = self.split_heads(K)
    V = self.split_heads(V)

    # Calculate Attention by heads
    attention_heads, attention_probabilities = self.attention(Q, K, V)
    self.attention_probabilities = attention_probabilities

    # Concatenate heads --> H
    attention_heads = self.combine_heads(attention_heads)

    # Pass H through W_o
    return self.W_o(attention_heads)

# ANN
## Projections block for into vector
class ProjectionInfo(torch.nn.Module):
  def __init__(self, info_depth: int, depth: int):
    super(ProjectionInfo, self).__init__()
    self.fc1 = torch.nn.Linear(info_depth, 128)
    self.fc2 = torch.nn.Linear(128,64)
    self.fc3 = torch.nn.Linear(64,depth)
    self.relu = torch.nn.functional.relu
    # TODO: experiment with dropouts
  
  def forward(self, x):
    '''
    x: (batch_size, 1, info_depth)
    '''
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    return self.fc3(x)

# FFN Block of transformer
class FFN(torch.nn.Module):
  def __init__(self, depth: int, ff_dim=2048):
    super(FFN, self).__init__() 
    self.fc1 = torch.nn.Linear(depth, ff_dim)
    self.fc2 = torch.nn.Linear(ff_dim, 128)
    self.fc3 = torch.nn.Linear(128, 64)
    self.fc4 = torch.nn.Linear(64, 1)
    self.relu = torch.nn.functional.relu
  
  def forward(self, x):
    '''
    x: (batch_size, rc, depth)
    '''
    x = self.fc1(x)
    x = self.relu(x)
    x = self.fc2(x)
    x = self.relu(x)
    x = self.fc3(x)
    x = self.relu(x)
    return self.fc4(x) # (batch_size,1)
  
# TransformerBlock
class ChessTransformerClassification(torch.nn.Module):
  def __init__(self,depth:int=7, info_depth:int=13,positional_embedding_mode:str="periodic",n_heads=1):
    super(ChessTransformerClassification, self).__init__()
    
    self.info_depth = info_depth
    self.positional_embedding_mode = positional_embedding_mode
    self.n_heads = n_heads
    if self.positional_embedding_mode == "periodic":
      self.depth = depth
    elif self.positional_embedding_mode == "fixed":
      self.depth = depth + 1
    else:
      assert False, f"Expected 'periodic' or 'fixed', got invalid mode {self.positional_embedding_mode}!"

    self.MHA = MHA(self.depth, n_heads)
    self.FFN = FFN(self.depth)
    self.ProjectionInfo = ProjectionInfo(info_depth, self.depth)
    self.PE = ChessPositionalEncoding(self.depth, mode=positional_embedding_mode)
    
  def forward(self, batch_board, batch_info):
    '''
    batch_board: (batch_size, depth, row, col)
    batch_info: (batch_size, 1, info_depth)
    '''
    batch_size, depth, row, col = batch_board.shape
    rc = row*col
    batch_board = batch_board.reshape(batch_size,depth,rc) # (batch_size, depth, r,c) -> (batch_size, depth, rc)
    batch_board = batch_board.transpose(1,2) # (batch_size, depth, rc) -> (batch_size, rc, depth)
    # TODO: Implement CLS token concatenation
    batch_info = batch_info.unsqueeze(1)

    # TODO: Fix positional Embedding problem when using passing ProjectionInfo(batch_info) into PE which clases due to max_tokens arg
    q = self.ProjectionInfo(batch_info)
    k = v = self.PE(batch_board)
    o = self.MHA(q,k,v)
    o = self.FFN(o)
    o = torch.sigmoid(o)
    o = o.squeeze(2)
    return o