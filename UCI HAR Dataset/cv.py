import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

def cvTrain(X, y, Model, params, k=5, epochs=150, verbose=False):

    models, accuracies = [], []
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    
    if issubclass(Model, nn.Module):
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        batch_size=128
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
        
            train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
            valid_loader = DataLoader(Subset(dataset, valid_idx), batch_size=batch_size, shuffle=False)
            model = Model(**params).to('cuda')
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.NAdam(model.parameters(), lr=1e-3)#, weight_decay=1e-2)
            
            for epoch in tqdm(range(epochs), desc=f'Fold{fold+1}'):
            
                train_losses, valid_losses = [], []
                train_correct, train_total = 0, 0
                valid_correct, valid_total = 0, 0
            
                # --------------------
                # Train
                # --------------------
                model.train()
                for train, labels in train_loader:
                    optimizer.zero_grad()
                    outputs = model(train.to('cuda')).to('cpu')
                    train_loss = criterion(outputs, labels)
                    train_loss.backward()
                    optimizer.step()
                    train_losses.append(train_loss.item())
            
                    # Accuracy
                    predicted = torch.argmax(outputs, dim=1)
                    train_correct += (predicted == labels).sum().item()
                    train_total += labels.size(0)
            
                train_acc = train_correct / train_total
            
                # --------------------
                # Test
                # --------------------
                model.eval()
                with torch.no_grad():
                    for test, labels in valid_loader:
                        outputs = model(test.to('cuda')).to('cpu')
                        valid_loss = criterion(outputs, labels)
                        valid_losses.append(valid_loss.item())
            
                        # Accuracy
                        predicted = torch.argmax(outputs, dim=1)
                        valid_correct += (predicted == labels).sum().item()
                        valid_total += labels.size(0)
            
                valid_acc = valid_correct / valid_total
        
                if verbose & ((epoch+1) * 10 % epochs == 0):
                    print(f"Train: Epoch [{epoch+1}/{epochs}], Loss: {torch.tensor(train_losses).mean():.4f}, Accuracy: {train_acc:.4f}| Valid: Epoch [{epoch+1}/{epochs}], Loss: {torch.tensor(valid_losses).mean():.4f}, Accuracy: {valid_acc:.4f}")
                
            models.append(model)
            accuracies.append(valid_acc)
        print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

    elif issubclass(Model, BaseEstimator):        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]
        
            # 2D に変換して学習
            model = Model(**params)
            model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
        
            # 予測と精度計算
            y_pred = model.predict(X_valid.reshape(X_valid.shape[0], -1))
            acc = accuracy_score(y_valid, y_pred)
            models.append(model)
            accuracies.append(acc)
            
            if verbose: print(f"Fold {fold+1} Accuracy: {acc:.4f}")
        
        print(f"Mean Accuracy: {np.mean(accuracies):.4f}")

    else: print("model must be torch.nn.Module or sklearn")
    return models

def cvTest(X, y, models, verbose=False):
    if all(isinstance(model, nn.Module) for model in models): 
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(dataset, batch_size=128, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        preds, fold = [], 0
        for model in tqdm(models):
            model.eval()
            test_pred, test_losses = [], []
            with torch.no_grad():
                for test, labels in loader:
                    outputs = model(test.to('cuda')).to('cpu')
                    test_pred.append(outputs)
                    test_loss = criterion(outputs, labels)
                    test_losses.append(test_loss.item())
                    
            preds.append(torch.cat(test_pred))
            fold += 1
            if verbose: print(f"Fold [{fold}], Loss: {torch.tensor(test_losses).mean():.4f}, Accuracy: {accuracy_score(y, torch.cat(test_pred).argmax(dim=1)):.4f}")
            
        y_pred = torch.stack(preds, dim=0).mean(dim=0).argmax(dim=1).cpu().numpy()
        print(f"Mean Accuracy: {accuracy_score(y, y_pred):.4f}")
        
    elif all(isinstance(model, BaseEstimator) for model in models): 
        y_pred = np.array([model.predict_proba(X.reshape(X.shape[0], -1)) for model in models]).mean(axis=0).argmax(axis=1)#.astype(bool)
        print("Accuracy:", accuracy_score(y, y_pred))
        
    else: print("model must be torch.nn.Module or sklearn")
    return y_pred, accuracy_score(y, y_pred)