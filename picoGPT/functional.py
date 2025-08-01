import torch
import math

def softmax(x, dim=-1):
  """ computes softmax over a given dim """
  exponential = torch.exp(x)
  return exponential/torch.sum(exponential,dim=dim,keepdim=True) # (B,T,C) -> (B,T,C)

def cross_entropy(logits, targets):
  """
  computes cross entropy over a batch of sequences of token embeddings
  this represents the average of the negative log probabilities that the logits assign to each ground truth target token
  """
  assert len(logits.shape) == 3 and len(targets.shape) == 2
  B,T,C = logits.shape
  B_2,T_2 = targets.shape
  assert B == B_2 and T == T_2
  probs = softmax(logits,dim=-1) # (B,T,C)
  target_probs = torch.gather(probs,2,torch.unsqueeze(targets,2)) # (B,T,1), find the probability of the target according to the logits
  return -torch.mean(torch.log(target_probs)) # (B,T,1) -> (1,)

def gelu(x):
  """ computes gelu nonlinearity """
  return 0.5*x*(1+torch.tanh((2/math.pi)**0.5 * (x+ 0.044715* x**3))) # (B,T,C)