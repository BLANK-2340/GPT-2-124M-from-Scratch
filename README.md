# GPT-2 124M from Scratch

This repository contains a clean, from-scratch implementation of the GPT-2 124M language model using PyTorch.

## Features

-   **No External Dependencies**: Implemented without `transformers` or `tiktoken`. Only uses `torch`.
-   **Custom Tokenizer**: Includes a simple character-level tokenizer.
-   **Model Architecture**: Full implementation of GPT-2 architecture (Attention, MLP, LayerNorm, Block).
-   **Training & Generation**: Scripts for training on custom text and generating samples.

## Files

-   `model.py`: GPT-2 model definition.
-   `train.py`: Training loop and data loading.
-   `generate.py`: Text generation script.
-   `tokenizer.py`: Custom tokenizer implementation.
-   `config.py`: Model configuration.
-   `verify.py`: Script to verify parameter count and forward pass.

## Usage

1.  **Train the model**:
    ```bash
    python train.py
    ```

2.  **Generate text**:
    ```bash
    python generate.py
    ```

3.  **Verify implementation**:
    ```bash
    python verify.py
    ```

## Requirements

-   Python 3.8+
-   PyTorch
