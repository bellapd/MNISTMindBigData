import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from src.classifier.eval import evaluate_model


class CNN(nn.Module):
    def __init__(self, n_ch, n_times, n_classes, F1=8, D=2, F2=16, kern_len=64, dropout_rate=0.25):
        super().__init__()

        self.tempConv = nn.Conv2d(1, F1, (1, kern_len), padding=(0, kern_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthConv = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.elu = nn.ELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        self.sepConv = nn.Conv2d(F1 * D, F2, (1, 16), padding=(0, 8), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        t_out = n_times // 4 // 8
        self.classify = nn.Linear(F2 * t_out, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.tempConv(x)
        x = self.bn1(x)
        x = self.depthConv(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        x = self.sepConv(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        x = x.flatten(start_dim=1)
        x = self.classify(x)
        return x


def train_CNN_model(model_class, cleaned_eeg_path, labels_path, model_weights_path, batch_size=32, lr=1e-4):
    X = np.load(cleaned_eeg_path)
    y = np.load(labels_path)

    X = torch.from_numpy(X).float().unsqueeze(1)
    y = torch.from_numpy(y).long()

    # Normalize each trial
    X = X - X.mean(dim=-1, keepdim=True)
    X = X / (X.std(dim=-1, keepdim=True) + 1e-8)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_trials, _, n_ch, n_times = X.shape
    n_classes = int(y.max().item()) + 1

    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=0.2, stratify=y.numpy(), random_state=42
    )
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).long()
    y_test = torch.from_numpy(y_test).long()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    model = model_class(n_ch=n_ch, n_times=n_times, n_classes=n_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(1, 201):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss += criterion(model(xb), yb).item()
        val_loss /= len(test_loader)

        print(f"Epoch {epoch:03d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_weights_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping.")
                break

    print("\nEvaluating best model on test set:")
    model.load_state_dict(torch.load(model_weights_path))
    evaluate_model(model, test_loader, device, model_name="CNN")