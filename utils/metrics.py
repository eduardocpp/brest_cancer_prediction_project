from sklearn.metrics import accuracy_score, confusion_matrix

def evaluate_predictions(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=["B", "M"])
    return acc, cm
