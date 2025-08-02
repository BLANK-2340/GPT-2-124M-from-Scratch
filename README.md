GPT‑2 124M From Scratch
=======================

This repository contains a simple example project that demonstrates how to train a medium‑sized [GPT‑2](https://openai.com/blog/better-language-models) language model from scratch using the [`transformers`](https://github.com/huggingface/transformers) library from Hugging Face.  The goal of this project is educational – it shows how to configure a GPT‑2 model with roughly 124 million parameters, build a training dataset, and run a basic training loop.  It is **not** intended to reproduce state‑of‑the‑art results or train a fully converged model on a massive corpus.  Instead it provides a clean starting point you can adapt for your own data and compute budget.

## Project overview

The repository consists of the following key files:

* **`train.py`** – A standalone Python script that builds a GPT‑2 model from scratch and fine‑tunes it on a user‑selected dataset.  You can specify the dataset (e.g., *wikitext* or a local text file), adjust hyper‑parameters such as the number of layers, attention heads, embedding size, sequence length, number of training epochs, and batch size via command‑line arguments, and save the trained model to disk.
* **`requirements.txt`** – Lists the Python dependencies needed to run the script.  These include PyTorch, Hugging Face `transformers` and `datasets`, and the `tqdm` progress bar library.

The default configuration in `train.py` creates a 12‑layer GPT‑2 model with a 768‑dimensional hidden size and 12 attention heads, which yields approximately 124 million parameters – similar to the GPT‑2 small model published by OpenAI.  The script uses Hugging Face’s `Trainer` API to handle the training loop and logging.

## Quick start

### Prerequisites

You need Python 3.8 or later installed along with `pip` for package management.  To set up the environment, clone this repository and install the dependencies:

```bash
git clone https://github.com/BLANK-2340/GPT-2-124M-from-Scratch.git
cd GPT-2-124M-from-Scratch
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, replace the last two lines with the appropriate virtual‑environment activation command (`.venv\Scripts\activate`).

### Training on an open dataset

By default, `train.py` loads the [WikiText‑2](https://paperswithcode.com/dataset/wikitext-2) dataset via the Hugging Face `datasets` library and trains the model for one epoch on sequences of 128 tokens.  You can run this default configuration with:

```bash
python train.py
```

The script will download the dataset and tokenizer, initialise a fresh GPT‑2 model from configuration, and begin training.  Model checkpoints and the final model weights will be written to `./model_output`.  Feel free to change `--epochs`, `--batch_size`, `--block_size`, and `--output_dir` to suit your needs.

### Training on your own text

If you have a corpus stored in a plain‑text file, you can train on it by passing `--train_file` along with the path to the text.  For example:

```bash
python train.py --train_file path/to/my_corpus.txt --epochs 3 --block_size 256
```

When using a custom text file, the script creates a simple streaming dataset that reads the file line by line and tokenizes it on the fly.  Make sure your corpus is large enough to benefit from the capacity of the GPT‑2 architecture – a few megabytes at minimum.

### Hyper‑parameters

Key hyper‑parameters can be adjusted via command‑line options:

* `--n_layers` – Number of transformer layers (default: 12).
* `--n_heads` – Number of attention heads per layer (default: 12).
* `--n_embd` – Dimensionality of the hidden state (default: 768).
* `--block_size` – Maximum sequence length; the context size of the model.
* `--batch_size` – Per‑device batch size.  Note that the effective batch size is `batch_size × number of GPUs` when using multiple GPUs.
* `--epochs` – Number of passes over the training set.

Adjusting these values will change the parameter count and memory footprint of the model.  For example, reducing `n_layers` from 12 to 6 produces a smaller, faster model at the cost of performance.  Conversely, increasing `n_embd` to 1024 and `n_layers` to 24 will produce a model closer to GPT‑2 medium.

## Caveats

1. **Compute requirements** – Training a GPT‑2 124M model on a large corpus requires a modern GPU with at least 8 GB of memory.  CPU training will be extremely slow.  Consider using gradient accumulation or reducing the batch size if you encounter memory issues.
2. **Dataset quality** – The model learns whatever patterns exist in the training data.  Poor‑quality or biased text will produce correspondingly poor results.  Always curate your dataset carefully.
3. **Experimentation** – Feel free to modify the script to explore different tokenizers, optimizer settings, or learning rate schedules.  The `transformers` library makes such experimentation straightforward.

## License

The code in this repository is released under the MIT license.  See [LICENSE](LICENSE) for details.