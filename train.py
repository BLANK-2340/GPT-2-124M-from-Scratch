import torch
import torch.optim as optim
from config import GPT2Config
from model import GPT2
from tokenizer import CharacterTokenizer
import os

# Hyperparameters
batch_size = 8
block_size = 128 # Context length
max_iters = 50
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create dummy data if not exists
if not os.path.exists('input.txt'):
    print("Creating dummy input.txt...")
    with open('input.txt', 'w', encoding='utf-8') as f:
        f.write("Hello world! This is a test dataset for training GPT-2 from scratch. " * 500)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize tokenizer
tokenizer = CharacterTokenizer()
tokenizer.train(text)
vocab_size = tokenizer.vocab_size
print(f"Vocab size: {vocab_size}")

# Encode data
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Initialize model
# Using default config but overriding vocab_size to match our simple tokenizer
config = GPT2Config(vocab_size=vocab_size, n_positions=block_size)
model = GPT2(config)
model.to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

model.train()
for iter in range(max_iters):
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if iter % 10 == 0:
        print(f"step {iter}: loss {loss.item():.4f}")

print("Training finished!")
torch.save(model.state_dict(), 'model.pt')
print("Model saved to model.pt")
