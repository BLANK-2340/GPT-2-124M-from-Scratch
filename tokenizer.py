class CharacterTokenizer:
    """
    A simple character-level tokenizer.
    """
    def __init__(self):
        self.chars = []
        self.stoi = {}
        self.itos = {}
        self.vocab_size = 0

    def train(self, text):
        """
        Builds the vocabulary from the given text.
        """
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        print(f"Tokenizer trained. Vocab size: {self.vocab_size}")

    def encode(self, text):
        """
        Converts a string to a list of integers.
        """
        return [self.stoi[c] for c in text]

    def decode(self, ids):
        """
        Converts a list of integers back to a string.
        """
        return ''.join([self.itos[i] for i in ids])

    def save(self, path):
        # Implementation for saving tokenizer state if needed
        pass

    def load(self, path):
        # Implementation for loading tokenizer state if needed
        pass
