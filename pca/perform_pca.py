from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd

def run_pca(X, y, n_components=10):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    explained_variance = pca.explained_variance_ratio_
    return X_pca, explained_variance, pca

def plot_pca_components(X_pca, y, components=(0, 1)):
    df = pd.DataFrame({
        "PC1": X_pca[:, components[0]],
        "PC2": X_pca[:, components[1]],
        "Label": y
    })
    plt.figure(figsize=(8,6))
    for label in df["Label"].unique():
        subset = df[df["Label"] == label]
        plt.scatter(subset["PC1"], subset["PC2"], label=label, alpha=0.5)
    plt.xlabel(f"PC{components[0]+1}")
    plt.ylabel(f"PC{components[1]+1}")
    plt.title("PCA - First Two Components")
    plt.legend()
    plt.tight_layout()
    plt.show()
