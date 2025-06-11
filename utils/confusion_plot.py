import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_confusion_matrix(cm, labels=["B", "M"], title="Matriz de Confusão"):
    """
    cm: matriz de confusão (array 2x2)
    labels: nomes das classes
    """
    df_cm = pd.DataFrame(cm, index=[f"Real {l}" for l in labels],
                             columns=[f"Previsto {l}" for l in labels])
    
    plt.figure(figsize=(6,4))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(title)
    plt.ylabel("Classe Real")
    plt.xlabel("Classe Predita")
    plt.tight_layout()
    plt.show()
