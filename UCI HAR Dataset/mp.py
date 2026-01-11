import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, Subset, DataLoader
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
import torch.multiprocessing as mp
import optuna
import random

# 設定
seed = 42
num_workers = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

def Train(model, train_loader, valid_loader, lr, q_acc, q_model, event, epochs):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)
    optim = torch.optim.NAdam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            optim.zero_grad()
            output = model(x.to(device))
            loss = criterion(output, y.to(device))
            loss.backward()
            optim.step()
            train_losses.append(loss.item())
        
        # 検証
        model.eval()
        valid_correct, valid_total = 0, 0
        valid_losses = []
        with torch.no_grad():
            for x, y in valid_loader:
                output = model(x.to(device))
                loss = criterion(output, y.to(device))
                valid_losses.append(loss.item())
                pred = torch.argmax(output, dim=1)
                valid_correct += (pred.cpu() == y).sum().item()
                valid_total += y.size(0)
        
        acc = valid_correct / valid_total
        q_acc.put(acc)  # 1 epochごとにAccuracyを送信
        event.clear()   # 次epochまで待機
        if not event.wait(timeout=3): break# cvTrainから許可が来るまで待機

    q_model.put(model.cpu())

def cvTrain(X, y, Model, params, k=5, epochs=150, lr=1e-3, trial=None):
    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    
    if issubclass(Model, nn.Module): 
        q_acc = mp.Queue()
        q_model = mp.Queue()
        event = mp.Event()
        processes = []
        
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))   
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y)):
            train_loader = DataLoader(
                    Subset(dataset, train_idx), batch_size=batch_size, shuffle=True,
                    worker_init_fn=seed_worker, generator=g, num_workers=num_workers
            )
            valid_loader = DataLoader(
                    Subset(dataset, valid_idx), batch_size=batch_size, shuffle=False,
                    worker_init_fn=seed_worker, generator=g, num_workers=num_workers
            )
        
            p = mp.Process(target=Train, args=(Model(**params),
                                               train_loader, valid_loader, lr,
                                               q_acc, q_model, event, epochs))
            p.start()
            processes.append(p)
            
        best, cnt = 0, 0
        for epoch in tqdm(range(epochs)):
            accuracies = [q_acc.get() for _ in range(k)]
            acc = np.mean(accuracies)

            if trial:
                trial.report(acc, epoch)
                if trial.should_prune(): 
                    print(f"========Optuna Pruning: Epoch{epoch+1}========")
                    break  
            if best>=acc: 
                cnt += 1
                if cnt>=epochs: 
                    print(f"========Early Stopping: Epoch{epoch+1}========")
                    break
            else:
                best = acc
                cnt = 0 
                    
            event.set()  # 次epochをTrainプロセスに許可
        models = [q_model.get() for _ in range(k)]
        for p in processes:
            p.terminate()        
        for p in processes:
            p.join()
            
    elif issubclass(Model, BaseEstimator): 
        res = cross_validate(Model(**params), X.reshape(X.shape[0], -1), y, cv=kf, n_jobs=k, scoring="accuracy", return_estimator=True)
        models = res['estimator']
        accuracies = res['test_score']

    print("Accuracy:", np.mean(accuracies))
    return models, accuracies

def Test(model, test_loader, queue):
    criterion = nn.CrossEntropyLoss()
    model = model.to(device)    
    model.eval()
    test_pred, test_losses = [], []
    with torch.no_grad():
        for test, labels in test_loader:
            outputs = model(test.to(device)).to('cpu')
            test_pred.append(outputs)
            test_loss = criterion(outputs, labels)
            test_losses.append(test_loss.item())
    queue.put(torch.cat(test_pred))

def cvTest(X, y, models, verbose=False):
    if all(isinstance(model, nn.Module) for model in models): 
        queue = mp.Queue()
        processes = []
        
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            worker_init_fn=seed_worker, generator=g, num_workers=num_workers
        )
        preds = []
        for model in tqdm(models):
            p = mp.Process(target=Test, args=(model, loader, queue))
            p.start()
            processes.append(p)
        for _ in models:
            preds.append(queue.get()) 
        for p in processes:
            p.join()
        y_pred = torch.stack(preds, dim=0).mean(dim=0).argmax(dim=1).cpu().numpy()
        print("Accuracy:", accuracy_score(y, y_pred))
        
    elif all(isinstance(model, BaseEstimator) for model in models): 
        y_pred = np.array([model.predict_proba(X.reshape(X.shape[0], -1)) for model in models]).mean(axis=0).argmax(axis=1)
        print("Accuracy:", accuracy_score(y, y_pred))
        
    else: print("model must be torch.nn.Module or sklearn")
    return y_pred, accuracy_score(y, y_pred)
