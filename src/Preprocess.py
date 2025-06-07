import pandas as pd
from sklearn.preprocessing import StandardScaler , OrdinalEncoder
from sklearn.model_selection import train_test_split

#function that removes irrelevant or noisy features
def remove(df):
    print(f'\033[95m[Preprocessing:]\033[0m')
    df.drop(columns=['Firewall Logs', 'IDS/IPS Alerts', 'Proxy Information', 'Payload Data', 'User Information' , 'Source Port'],
            inplace=True)
    print(
        f'\n\033[96m[columns dropped due to irrelevancy or noise:]\033[0m Firewall Logs , IDS/IPS Alerts , Proxy Information , Payload Data , User Information' , 'Source Port')

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

        if n == 'Malware Indicators':
            df[n] = df[n].fillna('Not Detected')
        elif n == 'Alerts/Warnings':
            df[n] = df[n].fillna('Not Triggered')

    print(f'\033[96m[Null values filled]\033[0m')
    print(df.isnull().sum())

    return df


# function that extracts features from the timestamp column
def timestamp(df):
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['month'] = df['Timestamp'].dt.month
    df['day'] = df['Timestamp'].dt.dayofweek
    df['hour'] = df['Timestamp'].dt.hour

    df.drop(columns=['Timestamp'] , inplace=True)
    return df


# function that categorizes Destination port
def port(df):
    def bin(port):
        if port <= 1023:
            return 'system'
        elif port < 49152:
            return 'registered'
        else:
            return 'dynamic'

    df['Destination Port'] = df['Destination Port'].apply(bin)
    return df


# function that splits the dataframe
def split(df):
    X = df.drop(columns=['Attack Type'])
    y = df['Attack Type']

    X_train, X_test, y_train, y_test = train_test_split(X , y , test_size=0.2 , random_state=42 , stratify=y)

    return X_train, X_test, y_train, y_test


# function that allows the user to choose the target column then applies normalization(standardization)
def normalization(X_train , X_test):
    numerical = ['Packet Length' , 'Anomaly Scores' , 'month' , 'day' , 'hour']

    scaler = StandardScaler()
    X_train[numerical] = scaler.fit_transform(X_train[numerical])
    X_test[numerical] = scaler.fit_transform(X_test[numerical])

    print('\033[91m[Normalization (Standardization) done]]\033[0m')
    print(X_train[numerical].describe().round(3))

    return X_train, X_test


# function that applies several encoding techniques
def encoding(X_train, X_test):
    freq = ['Geo-location Data' , 'Source IP Address' , 'Destination IP Address']
    one_hot = ['Destination Port' , 'Network Segment' ,  'OS' , 'Protocol' , 'Packet Type' , 'Log Source' , 'Traffic Type' , 'Malware Indicators' , 'Attack Signature' , 'Action Taken' , 'Alerts/Warnings']

    order = [['Low', 'Medium', 'High']]
    Oencoder = OrdinalEncoder(categories=order)
    X_train['Severity Level'] = Oencoder.fit_transform(X_train[['Severity Level']])
    X_test['Severity Level'] = Oencoder.fit_transform(X_test[['Severity Level']])

    for c in freq:
        frequ = X_train[c].value_counts(normalize=True)
        X_train[c + '_freq'] = X_train[c].map(frequ)
        X_test[c + '_freq'] = X_test[c].map(frequ)
        X_train.drop(columns=[c], inplace=True)
        X_test.drop(columns=[c], inplace=True)


    X_train = pd.get_dummies(X_train , columns=one_hot)
    X_test = pd.get_dummies(X_test , columns=one_hot)


    return X_train, X_test










