import seaborn as sns
from django.db.models.expressions import result
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score, f1_score


#function that fits the training sets to the models then evaluates
def fit(X_train , X_test , y_train , y_test):

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n\033[1m{name}\033[0m")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        print(classification_report(y_test, y_pred))

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'roc_auc': roc_auc_score(y_test, y_prob, multi_class='ovr')
        }

    return results


