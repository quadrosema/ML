import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


#reads and shows some facts about the data + partial DEA
def load(path):
    df = pd.read_csv(path)

    print(f'\n\033[96m[first 5 rows:]\033[0m')
    with pd.option_context('display.max_columns', None, 'display.width', None):
        print(df.head(20))

    print(f'\n\033[96m[information about the dataset:]\033[0m')
    df.info()

    print(f'\n\033[96m[description and stats:]\033[0m')
    print(df.describe())

    sns.countplot(x='stroke', data=df)
    plt.title("Imbalance data")
    plt.show()

    print('\033[91m' + '-' * 470 + '\033[0m\n')
    return df