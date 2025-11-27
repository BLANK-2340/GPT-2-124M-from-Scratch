import torch
from config import GPT2Config
from model import GPT2
from tokenizer import CharacterTokenizer
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load tokenizer (re-train on same data to ensure same vocab mapping)
if not os.path.exists('input.txt'):
    print("Error: input.txt not found. Run train.py first.")
    exit()

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = CharacterTokenizer()
tokenizer.train(text)

# Initialize model
config = GPT2Config(vocab_size=tokenizer.vocab_size, n_positions=128) # Must match train config
model = GPT2(config)
if os.path.exists('model.pt'):
    model.load_state_dict(torch.load('model.pt', map_location=device))
    print("Loaded trained model.")
else:
    print("No model found, using random weights.")
model.to(device)
model.eval()

# Generation
start_str = "Hello"
input_ids = tokenizer.encode(start_str)
x = torch.tensor([input_ids], dtype=torch.long, device=device)

print(f"Generating from prompt: '{start_str}'")

# Generate
max_new_tokens = 100
for _ in range(max_new_tokens):
    # Crop context if needed
    x_cond = x[:, -config.n_positions:]
    logits, _ = model(x_cond)
    logits = logits[:, -1, :] # Last time step
    probs = torch.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1)
    x = torch.cat((x, idx_next), dim=1)

generated_text = tokenizer.decode(x[0].tolist())
print("-" * 50)
print(generated_text)
print("-" * 50)
