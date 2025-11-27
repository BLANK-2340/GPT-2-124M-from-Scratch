import torch
from config import GPT2Config
from model import GPT2

def verify_model():
    print("Verifying GPT-2 124M implementation...")
    # Use default config for parameter count check (50257 vocab)
    config = GPT2Config() 
    model = GPT2(config)
    
    # 1. Parameter count check
    params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {params:,}")
    
    # Expected is around 124,439,808
    if 123_000_000 < params < 125_000_000:
        print("SUCCESS: Parameter count is correct (~124M).")
    else:
        print(f"WARNING: Parameter count {params} differs from expected ~124M.")

    # 2. Forward pass check
    print("Running forward pass check...")
    batch_size = 2
    seq_len = 128
    x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    try:
        logits, loss = model(x)
        print(f"Output shape: {logits.shape}")
        if logits.shape == (batch_size, seq_len, config.vocab_size):
            print("SUCCESS: Forward pass output shape is correct.")
        else:
            print(f"FAILURE: Output shape mismatch. Expected {(batch_size, seq_len, config.vocab_size)}, got {logits.shape}")
    except Exception as e:
        print(f"FAILURE: Forward pass failed with error: {e}")

if __name__ == "__main__":
    verify_model()
