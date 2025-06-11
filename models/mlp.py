from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
import itertools

def train_mlp(X_train, y_train, size_options=[10, 30, 50], max_layers=3, cv=5, alpha=0.0001, max_iter=400, random_state=42):
    # Gera combinações de 1 até max_layers camadas com os tamanhos fornecidos
    layer_configs = []
    for num_layers in range(1, max_layers + 1):
        combos = itertools.product(size_options, repeat=num_layers)
        layer_configs.extend(combos)

    best_score = 0
    best_model = None
    best_config = None

    for config in layer_configs:
        model = MLPClassifier(hidden_layer_sizes=config,
                              alpha=alpha,
                              max_iter=max_iter,
                              random_state=random_state)
        try:
            scores = cross_val_score(model, X_train, y_train, cv=cv)
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_config = config
        except:
            # Ignora modelos que não convergiram ou geraram erro
            continue
    
    if best_model is None:
        raise RuntimeError("No valid MLP configurations converged")

    best_model.fit(X_train, y_train)
    return best_model, best_config, best_score

def evaluate_mlp(model: MLPClassifier, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds