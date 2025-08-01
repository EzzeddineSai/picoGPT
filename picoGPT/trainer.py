
import math
import torch

class CosineScheduleWithWarmUp:
  """
  a learning rate scheduler that does a linear warm up and then a cosine decay between  max_lr and min_lr
  """
  def __init__(self, num_warmup_steps, num_total_steps, max_lr, min_lr):
    self.num_warmup_steps = num_warmup_steps
    self.num_total_steps = num_total_steps
    self.num_decay_steps = num_total_steps - num_warmup_steps # number of non-warm up steps
    self.max_lr = max_lr # max lr for non-warm up steps
    self.min_lr = min_lr # min lr for non-warm up steps
    self.current_step = 0 # stores how many steps the scheduler has run

  def step(self):
    self.current_step += 1

  def get_lr(self):
    if self.current_step <= self.num_warmup_steps and self.num_warmup_steps > 0:
      return self.max_lr*self.current_step/self.num_warmup_steps
    else:
      current_decay_step = self.current_step - self.num_warmup_steps
      return self.min_lr + (0.5*(self.max_lr-self.min_lr)*(1+math.cos(math.pi*current_decay_step/self.num_decay_steps)))
    
class AdamW:
  """
  an implementation of the AdamW optimizer
  it takes the parameters of the model and assigns to each tensor a first and second moment tensor that will be updated throughout
  """
  def __init__(self, params, init_lr=0, beta_1=0.9, beta_2=0.95, eps=1e-08, weight_decay=0):
    self.params = list(params)
    self.lr = init_lr
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.eps = eps
    self.weight_decay = weight_decay
    self.m = {}
    self.v = {}
    self.t = 0

    for param in self.params:
      if param.requires_grad:
        self.m[id(param)] = torch.zeros_like(param.data)
        self.v[id(param)] = torch.zeros_like(param.data)

  def step(self):
    """
    a gradient descend step is applied to the model parameters according to the AdamW algorithm
    uniquness is checked to make sure tied weights are not clipped twice
    """
    self.t += 1
    processed_ids = set()
    for param in self.params:
      if param.requires_grad and param.grad is not None:
        if id(param) not in processed_ids: # ensure parameter tensors are not updated twice
          self.m[id(param)] = self.beta_1 * self.m[id(param)] + (1 - self.beta_1) * param.grad # update first moment
          self.v[id(param)] = self.beta_2 * self.v[id(param)] + (1 - self.beta_2) * param.grad** 2 # update second moment
          m_hat = self.m[id(param)] / (1 - self.beta_1 ** self.t) # gives a less biased estimate of first moment
          v_hat = self.v[id(param)] / (1 - self.beta_2 ** self.t) # gives a less biased estimate of second moment
          param.data *= (1- (self.lr*self.weight_decay)) # weight decay
          param.data -= self.lr * m_hat/(torch.sqrt(v_hat)+self.eps) # main gradient descend step
          processed_ids.add(id(param))

  def clip_grad_norm(self, max_norm=1.0):
    """
    clip_grad_norm() computes the step's total L2 norm and if it is bigger than the max_norm it is scaled by the total norm
    uniquness is checked to make sure tied weights are not clipped twice
    """

    processed_ids = set()
    total_norm_squared = 0
    for param in self.params:
      if param.requires_grad and param.grad is not None:
        if id(param) not in processed_ids: # ensure parameter tensors are not updated twice
          total_norm_squared  += param.grad.data.norm().item() ** 2
          processed_ids.add(id(param))

    total_norm = total_norm_squared ** 0.5

    if total_norm > max_norm:
      processed_ids = set()
      for param in self.params:
        if param.requires_grad and param.grad is not None:
          if id(param) not in processed_ids: # ensure parameter tensors are not updated twice
            param.grad.data *= max_norm / total_norm
            processed_ids.add(id(param))

  def change_lr(self, lr):
    self.lr = lr

  def zero_grad(self):
    for param in self.params:
      param.grad = None


class Trainer:
    def __init__(self, model, num_total_steps, max_lr, min_lr, num_warmup_steps, beta_1, beta_2, weight_decay):
        self.model = model
        self.optimizer = AdamW(model.parameters(), init_lr=0, beta_1=beta_1, beta_2=beta_2,weight_decay=weight_decay)
        self.scheduler = CosineScheduleWithWarmUp(num_total_steps, num_total_steps, max_lr, min_lr)

    def step(self, xbatch, ybatch):
        current_learning_rate = self.scheduler.get_lr()
        logits, loss = self.model(xbatch,ybatch)
        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.change_lr(current_learning_rate)
        self.optimizer.clip_grad_norm()
        self.optimizer.step()
        self.scheduler.step()

        return loss.item(), current_learning_rate

