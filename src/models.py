from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    RocCurveDisplay,
)
import numpy as np

# function that fits the training sets to the models then evaluates
def fit(X_train, X_test, y_train, y_test):
    print(f"\n\033[95m[MODEL TRAINING STARTING:]\033[0m")

    models = {
        "Logistic Regression": LogisticRegression(
            solver="liblinear", penalty="l1", max_iter=100, C=0.01
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=5,
            class_weight="balanced",
            min_data_in_leaf=100,
        ),
        "Random Forest": RandomForestClassifier(
            criterion="gini", max_depth=1, n_estimators=100
        ),
        "XGBoost": XGBClassifier(
            eval_metric="aucpr",
            subsample=0.8,
            reg_lambda=1.5,
            reg_alpha=0,
            n_estimators=300,
            max_depth=7,
            learning_rate=0.2,
            gamma=0,
            colsample_bytree=0.8,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }


    results = {}

    for name, model in models.items():
        print(f"\n\033[1m{name}\033[0m")
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_threshold = thresholds[np.argmax(f1_scores)]
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        model.fit(X_train, y_train)
        RocCurveDisplay.from_estimator(model, X_test, y_test)
        plt.title(f"ROC curve for: {name}")
        plt.show()

    print(f"\n\033[95m[MODEL TRAINING END:]\033[0m")
    print("\033[91m" + "-" * 470 + "\033[0m\n")

    return results

