from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import numpy as np
from sklearn.linear_model import LogisticRegression as LR
from collections import Counter
from sklearn.model_selection import train_test_split


# function that applies SMOTE for oversampling
def balance(df):
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    print(f"\n\033[95m[PREPARATION FOR MODELS STARTED:]\033[0m")
    print(f"\n\033[96m[Before SMOTE:]\033[0m {Counter(y)}")

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)
    print(f"\033[96m[After SMOTE:]\033[0m {Counter(y)}")

    return X, y


# function that splits the dataframe
def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test


# function that applies several feature selection techniques then votes for best features
def selection(X_train, y_train, X_test):
    rfc = RFC(n_estimators=150, random_state=42, class_weight='balanced')
    rfc.fit(X_train, y_train)
    imp = rfc.feature_importances_
    thresh = np.mean(imp)
    rfc_features = set(X_train.columns[imp >= thresh])

    kbest = SelectKBest(score_func=f_classif, k=9)
    kbest.fit(X_train, y_train)
    kb_features = set(X_train.columns[kbest.get_support()])

    est = LR(max_iter=1000, random_state=42, solver='saga')
    rfe = RFE(estimator=est, n_features_to_select=9, step=0.1)
    rfe.fit(X_train, y_train)
    rfe_features = set(X_train.columns[rfe.support_])

    all_features = list(kb_features) + list(rfc_features) + list(rfe_features)
    votes = Counter(all_features)
    final = [f for f, c in votes.items() if c >= 2]

    X_train = X_train[final]
    X_test = X_test[final]

    print(f"\033[96m[Features selected:]\033[0m {final}")
    print(f"\033[95m[MODEL PREPARATION END:]\033[0m")
    print("\033[91m" + "-" * 470 + "\033[0m\n")

    return X_train, X_test, final

