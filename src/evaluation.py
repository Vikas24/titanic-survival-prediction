"""
Model evaluation module.

Includes:
- Classification metrics
- Confusion matrix visualization
- ROC-AUC analysis
- Precision-Recall analysis
- Cross-validation performance evaluation
"""


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

from sklearn.model_selection import cross_val_score


def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Evaluates model using multiple metrics and visualizations.
    """

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # -------------------------
    # Basic Metrics
    # -------------------------
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    # -------------------------
    # Confusion Matrix
    # -------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("../results/confusion_matrix.png")
    plt.show()

    # -------------------------
    # ROC Curve
    # ROC-AUC helps evaluate class separability independent of threshold
    # -------------------------
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig("../results/roc_curve.png")
    plt.show()

    # -------------------------
    # Precision–Recall Curve
    # Precision–Recall curve is more informative for imbalanced datasets
    # -------------------------
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    avg_precision = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP = {avg_precision:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.savefig("../results/precision_recall_curve.png")
    plt.show()

    # -------------------------
    # Precision–Recall vs Threshold
    # -------------------------
    plt.figure(figsize=(6, 4))
    plt.plot(thresholds, precision[:-1], label="Precision")
    plt.plot(thresholds, recall[:-1], label="Recall")
    plt.xlabel("Decision Threshold")
    plt.ylabel("Score")
    plt.title("Precision–Recall vs Threshold")
    plt.legend()
    plt.savefig("../results/precision_recall_threshold.png")
    plt.show()

    # -------------------------
    # Cross-Validation ROC-AUC
    # -------------------------
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="roc_auc"
    )

    print("Cross-validation ROC-AUC scores:", cv_scores)
    print("Mean ROC-AUC:", cv_scores.mean())
