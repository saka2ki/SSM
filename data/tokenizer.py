class Tokenizer:
  def __init__(self, texts):
      self.stoi = {ch: i for i, ch in enumerate(sorted(set(texts)))}
      self.itos = {i: ch for ch, i in self.stoi.items()}

  def encode(self, text):
      return [self.stoi[ch] for ch in text]

  def decode(self, tokens):
      return "".join(self.itos[token] for token in tokens.tolist())