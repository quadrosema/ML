import pandas as pd
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.model_selection import train_test_split


#function that removes irrelevant or noisy features
def remove(df):

    print(f'\033[95m[PREPROCESSING STARTED:]\033[0m')
    df.drop(columns=['id'] , inplace=True)
    print(f'\n\033[96m[columns dropped due to irrelevancy or noise:]\033[0m Doctor , Name , Room Number , Hospital')

    return df


# function for removing dupes and comparing before and after
def remove_dupes(df):
    before = df.shape[0]
    df = df.drop_duplicates()
    after = df.shape[0]
    dupe = before - after

    if dupe == 0:
        print(f'\n\033[96m[num of dupes:]\033[0m no duplicate rows')
    else:
        print(f'\n\033[96m[num of dupes:]\033[0m removed {dupe} duplicate rows.')

    return df


# function for handling null values according to their dtype
def null_handling(df):

    print(f'\n\033[96m[nulls:]\033[0m')
    print(df.isnull().sum().sort_values(ascending=False))
    nulls = []

    for c in df.columns:
        if df[c].isnull().any():
            nulls.append(c)

    for n in nulls:
        amount = df[n].isnull().sum()
        percent = (amount / len(df)) * 100
        print(f'\033[96m[column {n}:]\033[0m has {amount} nulls ({percent:.2f}%).')

        if n == 'bmi':
            df[n] = df[n].fillna(df[n].mean())
        elif n == 'smoking_status':
            df = df.drop(columns=n)

    print(f'\033[96m[Null values filled]\033[0m')
    print(df.isnull().sum())

    return df


# function that allows the user to choose the target column then applies normalization(standardization)
def normalization(df):

    numerical = ['age' , 'avg_glucose_level' , 'bmi']

    scaler = StandardScaler()
    df[numerical] = scaler.fit_transform(df[numerical])

    print('\033[91m[Normalization (Standardization) done]]\033[0m')

    return df


# function that applies several encoding techniques
def encoding(df):

    c = ['gender', 'ever_married', 'work_type', 'Residence_type']
    df = pd.get_dummies(df , columns=c)
    df.head()

    print('\033[96m[Encoding (One-hot) done]]\033[0m')
    print('\033[95m[PREPROCESSING DONE]]\033[0m')
    print('\033[91m' + '-' * 470 + '\033[0m\n')


    return df

