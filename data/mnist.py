from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class totensor:
  def __call__(self, pic):
    return torch.from_numpy(np.array(pic)).to(torch.long)
# データ変換の定義
transform = transforms.Compose([
    #transforms.ToTensor(), # PIL ImageをTensorに変換 (0-1に正規化される)
    #transforms.Normalize((0.1307,), (0.3081,)) # MNISTの平均と標準偏差で正規化
    totensor()
])

def MNIST():
    # MNISTデータセットのダウンロード
    train_dataset = datasets.MNIST(root='.', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='.', train=False, download=True, transform=transform)
    
    return train_dataset, test_dataset, 256