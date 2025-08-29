import torch
import random
from torch.utils.data import Dataset
from .tokenizer import Tokenizer

class InputCopyingDataset(Dataset):
    def __init__(self, length=5, n_samples=1024):
        self.keys = ["<BOS>", "<COPY>"] + [chr(i) for i in range(ord('a'), ord('z')+1)]
        self.tokenizer = Tokenizer(self.keys)
        self.length = length
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = random.sample(self.keys[2:], self.length)
        tokens = self.tokenizer.encode(
            ["<BOS>"] + sample + ["<COPY>"] + sample
        )
        return torch.tensor(tokens, dtype=torch.long), torch.zeros(self.length)

def InputCopying(length=5, n_train=1024, n_test=128):
    train_dataset = InputCopyingDataset(length=length, n_samples=n_train)
    test_dataset = InputCopyingDataset(length=length, n_samples=n_test)

    return train_dataset, test_dataset
