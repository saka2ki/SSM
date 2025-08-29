import torch
import random
from torch.utils.data import Dataset
from .tokenizer import Tokenizer

class AssociateRecallDataset(Dataset):
    def __init__(self, length=5, n_samples=512):
        self.dicts ={
            "<BOS>": 0,
            "<COPY>": 1
        } | {
            chr(code): str(i+2) for i, code in enumerate(range(ord('a'), ord('z') + 1))
        }
        self.keys = list(self.dicts.keys())
        self.tokenizer = Tokenizer(self.keys + [self.dicts[key] for key in self.keys])
        self.length = length
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        sample = random.sample(self.keys, self.length)
        tokens = self.tokenizer.encode(
            [item for key in sample for item in (key, self.dicts[key])]
        )
        return torch.tensor(tokens, dtype=torch.long), torch.zeros(self.length)

def AssociateRecall(length=5, n_train=512, n_test=64):
    train_dataset = AssociateRecallDataset(length=length, n_samples=n_train)
    test_dataset = AssociateRecallDataset(length=length, n_samples=n_test)

    return train_dataset, test_dataset
