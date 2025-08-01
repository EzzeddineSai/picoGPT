import torch
import torch.nn as nn
import math
from functional import softmax, cross_entropy, gelu

class Linear(nn.Module):
  """ a standard linear layer """
  def __init__(self, input_dim, output_dim, bias=True, mean=0, std=1.0):
    super().__init__()
    self.input_dim = input_dim
    self.output_dim = output_dim
    self.weight = nn.Parameter(torch.normal(mean=mean, std=std,size=(output_dim,input_dim)))
    self.bias = None
    if bias:
      self.bias = nn.Parameter(torch.zeros(output_dim))

  def forward(self,x):
    assert len(x.shape) == 3
    B,T,C = x.shape
    assert C == self.input_dim
    x_transpose = torch.transpose(x,1,2) # (B,C,T) = (B,input_dim,T)
    weight = self.weight.view(1,self.output_dim,self.input_dim)

    if self.bias is not None:
      bias = self.bias.view(1,self.output_dim, 1)
      activation_transpose = weight @ x_transpose + bias # (B,output_dim,T)
      return torch.transpose(activation_transpose,1,2) # (B,T,output_dim)
    else:
      activation_transpose = weight @ x_transpose # (B,output_dim,T)
      return torch.transpose(activation_transpose,1,2) # (B,T,output_dim)

class Embedding(nn.Module):
  """ an embedding table """
  def __init__(self, vocab_size, embedding_size, mean=0, std=1.0):
    super().__init__()
    self.vocab_size = vocab_size
    self.embedding_size = embedding_size
    self.weight = nn.Parameter(torch.normal(mean=mean, std=std,size=(vocab_size, embedding_size)))

  def forward(self,idx):
    return self.weight[idx] # (B,T) -> (B,T,embedding_size)


class Dropout(nn.Module):
  """ standard dropout """
  def __init__(self,pdrop=0.1):
    super().__init__()
    self.pdrop = pdrop

  def forward(self,x):
    if not self.training:
      return x
    else:
      mask =  (torch.rand(x.shape, device=x.device) > self.pdrop).to(x.dtype)*(1/(1-self.pdrop))
      return mask*x # (B,T,C) -> (B,T,C)

class LayerNorm(nn.Module):
  """ normalizes the last dim to have mean 0 and std 1"""
  def __init__(self,n_embd,eps=1e-05):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(n_embd)) # 1d trainable parameters. Set to 1 at the start to have gaussian stats
    self.beta = nn.Parameter(torch.zeros(n_embd)) # 1d trainable parameters. Set to 0 at the start to have gaussian stats

  def forward(self,x):
    x_mean = torch.mean(x,dim=-1, keepdim=True) # (B,T,1)
    x_var = torch.var(x, dim=-1,keepdim=True, unbiased=False) # (B,T,1)
    x = (x - x_mean)/torch.sqrt(x_var + self.eps) # (B,T,n_embed)
    return (x*self.gamma) + self.beta # (B,T,n_embed) -> (B,T,n_embed)

class SelfAttention(nn.Module):
  """
  given a batch of sequences of token embeddings, computes query and key vectors for each token
  attention is computed based on the query and key vectors, with masking to allow attention to only the current and previous tokens
  dropout is applied to attention followed by a value matrix that assigns a value vector to each token based on the attention
  """
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.key = Linear(config.n_embd, config.head_size, std=config.initializer_range)
    self.query = Linear(config.n_embd, config.head_size, std=config.initializer_range)
    self.value = Linear(config.n_embd, config.head_size, std=config.initializer_range)
    self.attn_dropout = Dropout(config.pdrop)
    self.register_buffer('causal_mask',torch.tril(torch.ones(self.config.n_positions,self.config.n_positions))) # lower triangle of ones

  def forward(self,x):
    assert len(x.shape) == 3
    B,T,C = x.shape
    assert C == self.config.n_embd and T <= self.config.n_positions # T can be any size <= config.n_positions
    k = self.key(x) # (B,T,head_size)
    q = self.query(x) # (B,T,head_size)
    v = self.value(x) # (B,T,head_size)

    W = q @ k.transpose(1,2) * 1/(self.config.head_size**0.5) # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
    W = W.masked_fill(self.causal_mask[:T,:T].view(1,T,T) == 0, float('-inf')) # crop the mask and fill in -inf for True
    W = softmax(W,dim=-1) # apply softmax. The -inf will be zeroed
    W = self.attn_dropout(W)
    return W @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)

class MultiHeadedAttention(nn.Module):
  """
  given a batch of sequences of token embeddings, first computes the "heads" many value vectors for each token based on the tokens before it
  concatenates the value vectors and uses that to compute an embedding vector via a linear projection layer
  this is followed by a dropout
  """
  def __init__(self, config):
    super().__init__()
    self.heads = nn.ModuleList([SelfAttention(config) for _ in range(config.n_head)])
    self.c_proj = Linear(config.n_embd, config.n_embd,std=config.initializer_range)
    self.resid_dropout = Dropout(config.pdrop)

  def forward(self,x):
    x = torch.cat([head(x) for head in self.heads], dim=-1) # each head_i(x) has shape (B,T,head_size). Concatenating results in (B,T,n_embd) shape
    x = self.c_proj(x) # (B,T,n_embd) -> (B,T,n_embd)
    return self.resid_dropout(x) # (B,T,n_embd)

class MLP(nn.Module):
  """ a standard MLP with GPT2 initilization and gelu nonlinearity """
  def __init__(self, config):
    super().__init__()
    self.c_fc = Linear(config.n_embd,config.n_inner,std=config.initializer_range)
    mlp_std = config.initializer_range/((2*config.n_layer)**0.5)
    self.c_proj = Linear(config.n_inner,config.n_embd,mean=0,std=mlp_std)
    self.mlp_dropout = Dropout(config.pdrop)

  def forward(self,x):
    x = self.c_fc(x) # (B,T,n_embd) -> (B,T,n_inner)
    x = gelu(x)
    x = self.c_proj(x) # (B,T,n_inner) -> (B,T,n_embd)
    return self.mlp_dropout(x) # (B,T,n_embd) -> (B,T,n_embd)

class DecoderBlock(nn.Module):
  """
  given a batch of sequences of token embeddings, it normalizes them and computes for each an embedding vector via multiheaded attention
  these embedding vectors are added to the original embeddings via a residual connection
  the combined embeddings go through a second residual connection with a connection where they are normalized and processed via an MLP
  """
  def __init__(self,config):
    super().__init__()
    self.c_attn = MultiHeadedAttention(config)
    self.mlp = MLP(config)
    self.ln_1 = LayerNorm(config.n_embd,eps=config.layer_norm_epsilon)
    self.ln_2 = LayerNorm(config.n_embd,eps=config.layer_norm_epsilon)

  def forward(self,x):
    x = x + self.c_attn(self.ln_1(x)) # (B,T,n_embd) + (B,T,n_embd) -> (B,T,n_embd)
    return x + self.mlp(self.ln_2(x)) # (B,T,n_embd) -> (B,T,n_embd)

class Transformer(nn.Module):
  """
  given a batch of sequences of token indexes, it computes for each a token embedding and a positional embedding
  the token and positional embeddings are added and passed through a dropout
  this is then passed through "n_layer" many decoder blocks and then normalized
  """
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.wte = Embedding(config.vocab_size,config.n_embd, std=config.initializer_range) # token embeddings
    self.wpe = Embedding(config.n_positions,config.n_embd, std=config.initializer_range) # positional embeddings
    self.embedding_dropout = Dropout(config.pdrop)
    self.h = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])
    self.ln_f = LayerNorm(config.n_embd)

  def forward(self,idx):
    assert len(idx.shape) == 2
    B,T = idx.shape
    token_embeddings = self.wte(idx) # (B,T,n_embd)
    pos_embeddings = self.wpe(torch.arange(T,dtype=torch.long,device=idx.device)) # (B,T,n_embd)
    x = self.embedding_dropout(token_embeddings + pos_embeddings.view(1,T,self.config.n_embd))
    for hidden in self.h:
      x = hidden(x) # (B,T,n_embd) -> (B,T,n_embd)
    return self.ln_f(x) # (B,T,n_embd)

class GPT(nn.Module):
  """
  given a batch of sequences of tokens, it passes them through a transformer to get embedding vectors for each token
  the embedding vectors are passed through a linear layer to get vocab logits for each token, even intermediate tokens in the sequences
  this linear layer's weight it tied to the token embedding layer of the transformer
  the logits represent the logprobabilities the model predicts for the next token
  loss is computed via cross entropy loss if target tokens are provided
  """
  def __init__(self, config, device='cpu'):
    super().__init__()
    self.config = config
    self.transformer = Transformer(self.config)
    self.lm_head = Linear(config.n_embd, config.vocab_size, bias=False) # model output layer
    self.lm_head.weight = self.transformer.wte.weight # tie the weights
    self.to(device) # load model to device

  def forward(self,idx,targets=None):
    assert len(idx.shape) == 2
    B,T = idx.shape
    assert T <= self.config.n_positions, "sequence length is too long" # T can be any size <= config.n_positions
    x = self.transformer(idx) # (B,T) -> (B,T,n_embd)
    logits = self.lm_head(x) # (B,T,n_embd) -> (B,T,vocab_size)

    if targets is None:
      loss = None
    else:
      loss = cross_entropy(logits,targets)
    return logits, loss # (B,T,vocab_size), (1,)


  @torch.no_grad()
  def generate_next_token(self, idx,top_k=50, temperature=1.0):
    """ given a batch of sequences of tokens and generates a new token index for each sequence by sampling the model logits """
    assert len(idx.shape) == 2 and top_k <= self.config.vocab_size
    B,T = idx.shape

    self.eval()
    if T > self.config.n_positions:
      logits, loss = self.forward(idx[:,-self.config.n_positions:]) # consider only the last chunk of text
    else:
      logits, loss = self.forward(idx)
    next_token_logits = logits[:,-1,:]/temperature # predictions for last token only
    topk_logits, topk_indices = torch.topk(next_token_logits, top_k, dim=-1) # (B,k), (B,k), compute the top k logit values
    probs = softmax(topk_logits, dim=-1) # (B,k), obtain a distribution over the topk tokens only
    samples_from_topk_indices = torch.multinomial(probs,num_samples=1) # (B,1), sample z in [0,...,k-1] according the probability distribution
    next_tokens = torch.gather(topk_indices,1,samples_from_topk_indices) # (B,1), topk_indices[z]
    return torch.cat((idx,next_tokens),dim=1) # (B,T) -> (B,T+1) concatenate growing the second dimension