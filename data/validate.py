import torch

def compute_val_loss(model, device, data_loader, batch_size):
  model.eval() # turns off dropout for example
  xbatch, ybatch = data_loader.get_batch(batch_size) # get a validation batch
  xbatch = xbatch.to(device)
  ybatch = ybatch.to(device)
  logits, loss = model(xbatch,ybatch)
  model.train() # return to train mode
  return loss.item()

def test_model_generation(model, device, max_new_tokens, decoder, start_token_id, top_k=50, temperature=1.0):
  """ generates a sequence of tokens from the model, starts with the start_token_id and generates max_new_tokens many tokens """
  model.eval()
  new_line_id = start_token_id
  idx = torch.tensor([[new_line_id]],dtype=torch.long, device=device)
  for _ in range(max_new_tokens):
    idx = model.generate_next_token(idx,top_k=top_k,temperature=temperature)
  sampled_sequence = idx[0].tolist() # take only the first element of the batch, and convert to list
  model.train()
  return decoder.decode(sampled_sequence)