import pandas as pd
from user_agents import parse

#reads and shows some facts about the data + partial DEA
def load(path):
    df = pd.read_csv(path)

    print(f'\n\033[96m[first 5 rows:]\033[0m')
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(df.head())

    print(f'\n\033[96m[information about the dataset:]\033[0m')
    df.info()

    print(f'\n\033[96m[description and stats:]\033[0m')
    print(df.describe())

    df['OS'] = df['Device Information'].apply(lambda x: parse(x).os.family)
    df = df.drop(columns=['Device Information'])
    print(df['OS'])
    print(f'\n\033[96m[Device Description column replaced with OS for noise reduction]\033[0m')

    print('\033[91m' + '-' * 470 + '\033[0m\n')
    return df
