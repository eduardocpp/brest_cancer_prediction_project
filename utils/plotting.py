import matplotlib.pyplot as plt

def plot_top_features(top_features, title="Top-10 feature importance"):
    """
    top_features — lista de tuplas (nome, importância)
    """
    names  = [name for name, _ in top_features][::-1]   # põe a mais importante no topo
    values = [val  for _,    val in top_features][::-1]

    _, ax = plt.subplots(figsize=(8, 5))
    ax.barh(names, values)
    ax.set_xlabel("Importance (Gini)")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

