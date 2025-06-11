import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

def plot_roc_curves(models_preds, y_test, title="Curvas ROC dos Modelos"):
    """
    models_preds: lista de tuplas (nome_modelo, probabilidade_positiva)
    y_test: rótulos reais
    """
    plt.figure(figsize=(8, 6))

    for name, probs in models_preds:
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Aleatório (AUC = 0.5)")
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

