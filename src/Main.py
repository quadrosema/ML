from src.Read import load
from src.Preprocess import remove_dupes , null_handling , normalization , encoding , remove , timestamp , port , split
import pandas as pd
from src.prepare import balance , selection
from src.models import fit
from sklearn.preprocessing import LabelEncoder

df = load('../Data/CS_Attacks.csv')
df = remove(df)
df = remove_dupes(df)
df = null_handling(df)
df = port(df)
df = timestamp(df)


X_train , X_test, y_train, y_test = split(df)
X_train , X_test = normalization(X_train , X_test)
X_train , X_test = encoding(X_train , X_test)
with pd.option_context('display.max_columns', None, 'display.width', None):
    print(X_train.head())

X_train , y_train = balance(X_train , y_train)
X_train , X_test , final = selection(X_train , y_train , X_test)


le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

model = fit(X_train , X_test , y_train , y_test)




