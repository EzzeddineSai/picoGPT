class GPTConfig:
  def __init__(self, vocab_size, model_downsize=1):
    assert model_downsize in [1,2,4], " model_downsize must be either 1, 2 or 4 "
    self.vocab_size = vocab_size
    self.model_downsize = model_downsize # makes the default GPT2 model smaller
    self.n_positions = 1024//model_downsize # context size
    self.n_embd = 12*64//model_downsize # embedding dimension
    self.n_head = 12//model_downsize # number of heads
    assert self.n_embd % self.n_head == 0
    self.head_size = self.n_embd//self.n_head # size of each head
    self.n_inner = self.n_embd*4 # MLP dimension
    self.n_layer = 12 # how many transformer block
    self.layer_norm_epsilon = 1e-5 # constant to avoid division by 0 in layer norm
    self.pdrop = 0.1 # dropout
    self.initializer_range = 0.02 # standard deviation for initializing weights