import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device, load_path=None, model_name="Model", save_path=None):
    """
    Evaluate a PyTorch model on the test set and print classification metrics and table summary.
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

    # === Metrics ===
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")

    # === Print classification report ===
    print(classification_report(y_true, y_pred))

    # === Confusion Matrix Plot ===
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    disp.ax_.set_title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        disp.figure_.savefig(save_path, dpi=300)
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

    # === Markdown-style summary table ===
    print("\n| Model | Accuracy | F1 Macro | Precision | Recall |")
    print("|-------|----------|----------|-----------|--------|")
    print(f"| {model_name} | {acc*100:.2f}% | {f1_macro:.4f} | {precision:.4f} | {recall:.4f} |")
