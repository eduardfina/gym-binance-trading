import os
import pandas as pd
import multiprocessing as mp


def load_dataset(name):
    print(f'Loading dataset {name}')
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.csv')
    df = pd.read_csv(str(path), parse_dates=True)
    return df
