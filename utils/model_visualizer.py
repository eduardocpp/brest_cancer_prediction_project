import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import plot_tree

def plot_model_architecture(model, name, X_pca, y):
    name = name.lower()

    if name == "decision tree":
        plt.figure(figsize=(15, 8))
        plot_tree(model, filled=True, fontsize=6, rounded=True)
        plt.title("Decision Tree Structure")
        plt.show()

    elif name == "mlp":
        # Representar a rede neural com círculos e conexões
        layer_sizes = [model.coefs_[0].shape[0]]  # entrada
        for coef in model.coefs_:
            layer_sizes.append(coef.shape[1])  # cada camada oculta + saída

        fig, ax = plt.subplots(figsize=(10, 4))
        for i, layer_size in enumerate(layer_sizes):
            for j in range(layer_size):
                circle = plt.Circle((i * 2, j * 1.5), radius=0.3, color='skyblue', ec='black')
                ax.add_patch(circle)
                ax.text(i * 2, j * 1.5, f'{j+1}', fontsize=6, ha='center', va='center')
            if i < len(layer_sizes) - 1:
                next_layer_size = layer_sizes[i + 1]
                for j in range(layer_size):
                    for k in range(next_layer_size):
                        ax.plot([i * 2, (i + 1) * 2], [j * 1.5, k * 1.5], color='gray', lw=0.5)

        ax.set_xlim(-1, 2 * len(layer_sizes))
        ax.set_ylim(-1, max(layer_sizes) * 1.5)
        ax.axis('off')
        plt.title("MLP Architecture")
        plt.show()
    elif name == "svm":
        plot_svm_support_vectors(model, X_pca, y)
    else:
        print(f"[INFO] Arquitetura não implementada para: {name}")
        print(model)


def plot_svm_support_vectors(model, X_pca, y, title="SVM Support Vectors (PC1 vs PC2)"):
    """
    Plota os vetores de suporte do modelo SVM com base nos dois primeiros PCs.
    
    Parâmetros:
        - model: modelo SVM treinado
        - X_pca: matriz transformada por PCA (espera-se shape [n_samples, 2+])
        - y: rótulos reais
    """
    if X_pca.shape[1] < 2:
        raise ValueError("É necessário ao menos 2 componentes principais (PCs) para visualização.")

    df = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Label": y
    })

    plt.figure(figsize=(8, 6))
    
    for label in df["Label"].unique():
        subset = df[df["Label"] == label]
        plt.scatter(subset["PC1"], subset["PC2"], label=f"Classe {label}", alpha=0.5)

    # Plotar vetores de suporte
    if hasattr(model, "support_"):
        sv_indices = model.support_
        plt.scatter(X_pca[sv_indices, 0], X_pca[sv_indices, 1],
                    facecolors='none', edgecolors='black', s=100,
                    label="Vetores de Suporte")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
