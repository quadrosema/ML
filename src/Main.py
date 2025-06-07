from src.Preprocess import null_handling
from src.Read import load
from src.Preprocess import remove_dupes , null_handling , normalization , encoding , remove , timestamp , port , split
import pandas as pd
from src.preparation import sampling , selection

df = load('../Data/CS_Attacks.csv')
df = remove(df)
df = remove_dupes(df)
df = null_handling(df)
df = port(df)
df = timestamp(df)


X_train , X_test, y_train, y_test = split(df)
X_train , X_test = normalization(X_train , X_test)
X_train , X_test = encoding(X_train , X_test)


X_train , X_test = sampling(X_train , X_test)
X_train , X_test = selection(X_train , y_train , X_test)




with pd.option_context('display.max_columns', None, 'display.width', None):
    print(X_train.head())
