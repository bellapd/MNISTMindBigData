import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.classifier.eval import evaluate_model  


class MLP(nn.Module):
    def __init__(self, input_dim: int, n_classes: int, dropout_rate: float = 0.3):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            # First hidden layer: 512 -> BatchNorm -> ReLU -> Dropout
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Second hidden layer: 256 -> BatchNorm -> ReLU -> Dropout
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Third hidden layer: 128 -> BatchNorm -> ReLU -> Dropout
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # Final linear layer: 128 -> n_classes
            nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_MLP_model(
    model_class,
    features_path: str,
    labels_path: str,
    model_weights_path: str,
    batch_size: int = 32,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 10,
):
    """
    Train an MLP on precomputed features (e.g., FFT or any other 1D feature vector).
    Saves the best‐performing model weights to `model_weights_path` and finally calls `evaluate_model(...)`.

    Parameters
    ----------
    model_class : class
        A subclass of nn.Module (e.g., MLP). Must accept `input_dim` and `n_classes` in its constructor.
    features_path : str
        Path to a .npy file of shape (n_trials, n_features).
    labels_path : str
        Path to a .npy file of shape (n_trials,).
    model_weights_path : str
        Where to save the best‐checkpoint (.pth).
    batch_size : int
        Batch size for DataLoader (both train and validation/test).
    epochs : int
        Maximum number of epochs to train.
    lr : float
        Initial learning rate for the AdamW optimizer.
    patience : int
        Number of epochs with no improvement in validation loss before early stopping.
    """
    X = np.load(features_path)   # shape: (n_trials, n_features)
    y = np.load(labels_path)     # shape: (n_trials,)

    # Z‐score normalization on each column (feature‐wise)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std  = X.std(axis=0, keepdims=True) + 1e-8
    X = (X - X_mean) / X_std

    # Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()   # dtype=torch.float32
    y_tensor = torch.from_numpy(y).long()    # dtype=torch.int64

    X_np = X_tensor.numpy()
    y_np = y_tensor.numpy()

    # Stratified split
    X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
        X_np, y_np, test_size=0.2, stratify=y_np, random_state=42
    )

    # Convert back to torch.Tensor
    X_train = torch.from_numpy(X_train_np).float()
    y_train = torch.from_numpy(y_train_np).long()
    X_test  = torch.from_numpy(X_test_np).float()
    y_test  = torch.from_numpy(y_test_np).long()

    # Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=batch_size,
        shuffle=False
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    input_dim = X_train.shape[1]
    n_classes = int(y_tensor.max().item()) + 1

    model = model_class(input_dim=input_dim, n_classes=n_classes).to(device)

    # Use AdamW with a bit of weight decay
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # Reduce LR by a factor of 0.5 if val_loss doesn’t improve for 5 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    counter = 0

    for epoch in range(1, epochs + 1):
        # --- Training Step ---
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validation Step ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                out = model(xb)
                batch_loss = criterion(out, yb).item()
                val_loss += batch_loss
                preds = out.argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        val_loss /= len(test_loader)
        val_acc = correct / total

        # Step the scheduler on validation loss
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc*100:.2f}%"
        )

        # Early stopping logic
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), model_weights_path)
            print(f"  → Saving new best model (val_loss={val_loss:.4f})")
        else:
            counter += 1
            if counter >= patience:
                print("  → Early stopping triggered")
                break

    print("\nEvaluating best model on the held-out test set:")
    model.load_state_dict(torch.load(model_weights_path))
    evaluate_model(model, test_loader, device, load_path=None, model_name="MLP")

    print("Done.\n")