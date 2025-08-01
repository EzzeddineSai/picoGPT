import numpy as np
import torch
import json

class Decoder():
  """
  takes the path to a vocab json file and a list of special tokens
  vocab file should be a dictionary with tokens as keys and ids as values
  """
  def __init__(self,vocab_path,special_tokens=[]):
    with open(vocab_path,'r') as f:
      self.token_to_id = json.load(f)
    assert self.token_to_id is not None and type(self.token_to_id) == dict, " vocab did not load correctly "

    self.id_to_token = {}
    for token,id in self.token_to_id.items():
      self.token_to_id[token] = id
      self.id_to_token[id] = token

    self.special_tokens_ids = []
    for token in special_tokens:
      self.special_tokens_ids.append(self.token_to_id[token])

  def decode(self,token_ids):
    """
    takes a list of token ids and returns the decoded string skipping special tokens
    """
    tokens = []
    for id in token_ids:
      if id in self.special_tokens_ids:
        continue
      tokens.append(self.id_to_token[id])
    return "".join(tokens)

class DataLoader():
  """
  loads batch_size many sequences of tokens from a bin file, each sequence n_pos many tokens
  the store should be a 1d numpy array of np.uint16 type
  """
  def __init__(self, data_set_path, n_positions):
    self.data_set_path = data_set_path
    self.n_positions = n_positions

  def get_batch(self, batch_size):
    data_source = np.memmap(self.data_set_path, dtype=np.uint16, mode='r')
    x_indexes_start = torch.randint(len(data_source)-self.n_positions-1,(batch_size,)).view(batch_size,1) # start index for each batch
    x_indexes = x_indexes_start + torch.arange(self.n_positions).view(1,self.n_positions)
    y_indexes = x_indexes + 1 # the targets for each subsequence is the next token
    x = torch.from_numpy(data_source[x_indexes]).long()
    y = torch.from_numpy(data_source[y_indexes]).long()
    return x,y