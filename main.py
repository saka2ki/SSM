import hydra
from omegaconf import DictConfig
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm.auto import tqdm

@hydra.main(config_name="config", version_base=None, config_path="conf")
def main(cfg: DictConfig) -> None:
    wandb.init(
        project="myproject",
        #name=cfg.name,
        config=dict(cfg)
    )
    train_dataset, test_dataset, cfg.model.init.vocab_size = hydra.utils.instantiate(cfg.data)
    model = hydra.utils.instantiate({"_target_": cfg.model._target_, **cfg.model.init}).to(cfg.device)
    device = cfg.device
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.bsz, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.bsz, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # 4. 学習サイクル
    print("--- 学習開始 ---")
    for epoch in range(cfg.epochs):
        train_losses, test_losses = [], []
        model.train()
        for train, labels in tqdm(train_loader):
            # 順伝播
            train = train.view(train.shape[0], -1).to(device)
            output = model(train[:, :-1], **cfg.model.forward)
    
            # 損失の計算
            train_loss = criterion(output.transpose(-2,-1), train[:, 1:])
            train_losses.append(train_loss.item())
    
            # 勾配のリセット
            optimizer.zero_grad()
    
            # 逆伝播
            train_loss.backward()
    
            # パラメータの更新
            optimizer.step()
            #break
            #wandb.log({"train_loss": train_loss})
            
        print(f"Train: Epoch [{epoch+1}/{cfg.epochs}], Loss: {torch.tensor(train_losses).mean():.4f}")
        model.eval()
        with torch.no_grad():
          for test, labels in tqdm(test_loader):
              # 順伝播
              test = test.view(test.shape[0], -1).to(device)
              output = model(test[:, :-1], **cfg.model.forward)
    
              # 損失の計算
              test_loss = criterion(output.transpose(-2,-1), test[:, 1:])
              test_losses.append(test_loss.item())
              #break
              #wandb.log({"test_loss": test_loss})
              
        #if epoch % 10 == 9:
        print(f"Test: Epoch [{epoch+1}/{cfg.epochs}], Loss: {torch.tensor(test_losses).mean():.4f}")

        wandb.log({
            "train_loss": torch.tensor(train_losses).mean(), 
            "test_loss": torch.tensor(test_losses).mean()
        })    
        torch.save(model.state_dict(), f'./module/{model.__class__.__name__}_{cfg.model.init.dim}.pth')
        wandb.save(f'./module/{model.__class__.__name__}_{cfg.model.init.dim}.pth')
    
    print("--- 学習終了 ---")

if __name__ == "__main__":
    main()