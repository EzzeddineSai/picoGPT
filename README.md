# picoGPT

An even more minimalist implementation of GPT2. Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT/). Focused on readability and clarity.

In addition GPT2, this includes a clear implementation of:

- PyTorch basic layers (Linear, LayerNorm, Dropout, Embedding)
- Pytorch functionals (softmax, cross_entropy, gelu)
- a data loader and a decoder
- AdamW with clipping, and a cosine scheduler with warmup

## Basic Usage

Having the repository and installed the requirements, you could easily initialize and train a model on 100k pages of wikipedia as follows:

```python 
from picoGPT.picoGPT_config import GPTConfig
from picoGPT.model import GPT
from picoGPT.trainer import Trainer

from data.wiki_100k import WikiLoader
from data.validate import compute_val_loss, test_model_generation

config = GPTConfig(vocab_size=10000,model_downsize=2)
dataset = WikiLoader(config)
train_data_loader, val_data_loader, decoder = dataset.get_loaders_and_decoder()

device = 'cuda' # or 'cpu'
my_model = GPT(config, device)
my_model = my_model.to(device)

num_total_steps = 1000 # or any other desired number
validation_steps = int(num_total_steps/10)
max_lr = 1e-4
min_lr = 2e-6
num_warmup_steps = int(num_total_steps/100)
weight_decay = 0.1
beta_1 = 0.9 
beta_2 = 0.95

lrs = []
losses = []

my_model.train()
for step in range(num_total_steps):
  
  xbatch, ybatch = train_data_loader.get_batch(batch_size)
  xbatch = xbatch.to(device)
  ybatch = ybatch.to(device)

  loss, current_learning_rate = trainer.step(xbatch,ybatch)

  lrs.append(current_learning_rate)
  losses.append(loss)

  if step%validation_steps == 0:
    val_loss = compute_val_loss(my_model, device, val_data_loader, batch_size)
    print(f'\nstep: {step}, loss: {loss}, validation loss: {val_loss}, current lr: {current_learning_rate}\n')
print(f'\nstep: {step}, loss: {loss}, validation loss: {val_loss}')
```

WHich should result in a validation loss that looks something like this:
![alt text](/assets/val_loss.png)

