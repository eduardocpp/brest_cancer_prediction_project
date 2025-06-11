from data.load_data import load_and_prepare_data
from pca.perform_pca import run_pca, plot_pca_components
from models.logistic_regression import train_logistic_regression, evaluate_logistic
from models.knn import train_knn, evaluate_knn
from models.random_forest import train_random_forest, evaluate_rf, get_feature_importance
from models.loess_approx import train_loess_approx, evaluate_loess
from models.svm import train_svm, evaluate_svm
from models.decision_tree import train_decision_tree, evaluate_decision_tree
from models.mlp import train_mlp, evaluate_mlp
from models.gradient_boosting import train_gradient_boosting, evaluate_gradient_boosting
from ensemble.majority_vote import majority_vote
from utils.metrics import evaluate_predictions
from utils.feature_analysis import summarize_top_features
from utils.plotting import plot_top_features
from utils.confusion_plot import plot_confusion_matrix
from utils.accuracy_plot import plot_model_accuracies
from utils.roc_plot import plot_roc_curves
from utils.model_visualizer import plot_model_architecture
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Carregar dados
X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()

# Transformar rótulos em binários
le = LabelEncoder()
y_test_bin = le.fit_transform(y_test)

# PCA
X_all = np.vstack([X_train, X_test])
y_all = pd.concat([y_train, y_test]).reset_index(drop=True)
X_all_pca, var_exp, pca_model = run_pca(X_all, y_all, n_components=10)
X_train_pca = X_all_pca[:len(X_train)]
X_test_pca = X_all_pca[len(X_train):]

# Treinar modelos
model_glm = train_logistic_regression(X_train_pca, y_train)
model_knn = train_knn(X_train_pca, y_train, k=5)
model_rf = train_random_forest(X_train_pca, y_train, mtry=5)
model_loess = train_loess_approx(X_train_pca, y_train)
model_svm = train_svm(X_train_pca, y_train)
model_decision_tree = train_decision_tree(X_train_pca, y_train)
model_mlp, config, score = train_mlp(X_train_pca, y_train)
model_gradient_boosting = train_gradient_boosting(X_train_pca, y_train)

# Avaliar modelos
acc_glm, pred_glm = evaluate_logistic(model_glm, X_test_pca, y_test)
acc_knn, pred_knn = evaluate_knn(model_knn, X_test_pca, y_test)
acc_rf, pred_rf = evaluate_rf(model_rf, X_test_pca, y_test)
acc_loess, pred_loess = evaluate_loess(model_loess, X_test_pca, y_test)
acc_svm, pred_svm = evaluate_svm(model_svm, X_test_pca, y_test)
acc_decision_tree, pred_decision_tree = evaluate_decision_tree(model_decision_tree, X_test_pca, y_test)
acc_mlp, pred_mlp = evaluate_mlp(model_mlp, X_test_pca, y_test)
acc_gradient_boosting, pred_gradient_boosting = evaluate_gradient_boosting(model_gradient_boosting, X_test_pca, y_test)

# Ensemble
ensemble_pred = majority_vote(pred_glm, pred_knn, pred_rf, pred_loess,
                               pred_svm, pred_decision_tree, pred_mlp, pred_gradient_boosting)
acc_ensemble, cm_ensemble = evaluate_predictions(y_test, ensemble_pred)

# Feature importance
top_features = get_feature_importance(model_rf, [f"PC{i+1}" for i in range(10)])
category_summary = summarize_top_features(top_features)

# Resultados
print("Acurácia GLM:", round(acc_glm, 3))
print("Acurácia KNN:", round(acc_knn, 3))
print("Acurácia Random Forest:", round(acc_rf, 3))
print("Acurácia Loess (Ridge approximation):", round(acc_loess, 3))
print("Acurácia SVM:", round(acc_svm, 3))
print("Acurácia Decision Tree:", round(acc_decision_tree, 3))
print("Acurácia MLP:", round(acc_mlp, 3))
print("Configuração MLP:", config, "Acurácia:", round(score, 3))
print("Acurácia Gradient Boosting:", round(acc_gradient_boosting, 3))
print("Acurácia Ensemble:", round(acc_ensemble, 3))
print("Matriz de confusão do Ensemble:")
print(cm_ensemble)
print("Top variáveis mais importantes (via PCA):", top_features)
print("Categorias no Top 10 (PCA):", category_summary)

# Plotar resultados
plot_pca_components(X_all_pca, y_all, components=(0, 1))
plot_top_features(top_features, title="Top-10 PCA Component Importance")

accuracies = {
    "GLM": acc_glm,
    "KNN": acc_knn,
    "Random Forest": acc_rf,
    "Loess": acc_loess,
    "SVM": acc_svm,
    "Decision Tree": acc_decision_tree,
    "MLP": acc_mlp,
    "Gradient Boosting": acc_gradient_boosting,
    "Ensemble": acc_ensemble
}
plot_model_accuracies(accuracies)
plot_confusion_matrix(cm_ensemble, labels=["B", "M"], title="Matriz de Confusão - Ensemble")

# Curvas ROC
models_with_proba = []

def get_model_probs(model, name):
    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba(X_test_pca)[:, 1]
            models_with_proba.append((name, probs))
        except:
            pass
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test_pca)
            models_with_proba.append((name, scores))
        except:
            pass

get_model_probs(model_glm, "GLM")
get_model_probs(model_knn, "KNN")
get_model_probs(model_rf, "Random Forest")
get_model_probs(model_svm, "SVM")
get_model_probs(model_decision_tree, "Decision Tree")
get_model_probs(model_mlp, "MLP")
get_model_probs(model_gradient_boosting, "Gradient Boosting")
get_model_probs(model_loess, "Loess (Ridge approx)")

plot_roc_curves(models_with_proba, y_test_bin)

plot_model_architecture(model_decision_tree, "decision tree", X_test_pca, y_test)
plot_model_architecture(model_mlp, "mlp", X_test_pca, y_test)
plot_model_architecture(model_svm, "svm", X_train_pca, y_train)

