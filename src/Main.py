from src.Read import load
from src.Preprocess import remove_dupes, null_handling, encoding, remove, normalization
from src.prepare import balance, selection, split
from src.models import fit

df = load('../Data/dataset.csv')
df = remove(df)
df = remove_dupes(df)
df = null_handling(df)


df = normalization(df)
df = encoding(df)

X , y = balance(df)
X_train , X_test, y_train, y_test = split(X , y)

X_train , X_test , final = selection(X_train, y_train , X_test)


results = fit(X_train , X_test , y_train , y_test)
print(results)


