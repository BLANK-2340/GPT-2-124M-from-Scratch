from dataclasses import dataclass

@dataclass
class GPT2Config:
    """
    Configuration class to store the configuration of a `GPT2Model`.
    """
    vocab_size: int = 50257 # Default GPT-2 vocab size, can be overridden
    n_embd: int = 768       # Embedding dimension
    n_layer: int = 12       # Number of transformer blocks
    n_head: int = 12        # Number of attention heads
    n_positions: int = 1024 # Maximum sequence length (context window)
    layer_norm_epsilon: float = 1e-5 # Epsilon for layer normalization
    dropout: float = 0.1    # Dropout probability


