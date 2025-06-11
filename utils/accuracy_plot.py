import matplotlib.pyplot as plt

def plot_model_accuracies(acc_dict, title="Acurácia dos Modelos"):
    """
    acc_dict: dicionário com nomes dos modelos como chaves e acurácias como valores
    """
    names = list(acc_dict.keys())
    values = list(acc_dict.values())

    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, values)
    plt.ylim(0.9, 1.0)
    plt.title(title)
    plt.ylabel("Acurácia")
    plt.xticks(rotation=15)

    # Adiciona valores acima das barras
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.002, f"{yval:.3f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show()
