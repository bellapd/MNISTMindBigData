import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, load_path=None, model_name="Model"):
    """
    Evaluate a PyTorch model on the test set and print classification metrics.
    
    Parameters
    ----------
    model : nn.Module
        Your trained PyTorch model instance.
    test_loader : DataLoader
        DataLoader containing test set.
    device : torch.device
        CPU or CUDA.
    load_path : str or None
        If given, loads model weights from file before evaluation.
    model_name : str
        Name of the model (for labeling plots/reports).
    """
    if load_path is not None:
        model.load_state_dict(torch.load(load_path))
        print(f"[{model_name}] Loaded weights from: {load_path}")
    
    model.to(device)
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true.extend(yb.numpy())
            y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    print(classification_report(y_true, y_pred))

    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    plt.show()

    print(f"\nEvaluation Results for {model_name}:")
    print(f"Accuracy     : {acc*100:.2f}%")
    print(f"F1 (Macro)   : {f1_macro:.4f}")
    print(f"F1 (Weighted): {f1_weighted:.4f}")