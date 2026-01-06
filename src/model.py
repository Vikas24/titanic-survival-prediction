"""
Model training module for Titanic Survival Prediction.

This module handles:
- Handling class imbalance using SMOTE
- Logistic Regression model training
- Hyperparameter tuning using GridSearchCV
"""


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE


def train_model(X_train, y_train):
    """
    Applies SMOTE and trains Logistic Regression using GridSearchCV.
    """
    # SMOTE is applied only on training data to handle class imbalance
    # Apply SMOTE on training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Logistic Regression + GridSearchCV
    # GridSearchCV is used to find optimal regularization parameters
    param_grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid.fit(X_train_smote, y_train_smote)

    print("✅ Best Hyperparameters:", grid.best_params_)

    return grid.best_estimator_
