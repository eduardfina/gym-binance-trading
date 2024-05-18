import os

import pandas as pd
import matplotlib.pyplot as plt

# Data paths
PATH = '../binance_trading_environment/datasets/data/crypto_data_1d.csv'
OUTPUT_DIR = 'images'

# Divide the data in 5 equal periods
PERIODS = [
    ['2017-08-27', '2019-01-28'],
    ['2019-01-29', '2020-07-10'],
    ['2020-07-11', '2021-12-21'],
    ['2021-12-22', '2023-06-03'],
    ['2023-06-04', '2024-03-05']
]

def generate_bitcoin_prices():
    data = pd.read_csv(PATH)
    data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')
    data.head()

    periods = [[pd.to_datetime(start), pd.to_datetime(end)] for start, end in PERIODS]

    # Filter by period and generate the graphs
    plt.figure(figsize=(15, 10))

    for i, (start, end) in enumerate(periods):
        period_data = data[(data['timestamp'] >= start) & (data['timestamp'] <= end)]

        plt.figure(figsize=(15, 8))
        plt.plot(period_data['timestamp'], period_data['BTC_close'], color='orange')
        plt.title(f'Període {i + 1}: {start.date()} a {end.date()}')
        plt.xlabel('Data')
        plt.ylabel('Preu de Bitcoin (USD)')
        plt.grid(True)

        # Save the image
        output_path = os.path.join(OUTPUT_DIR, f'BTC_price_period_{i + 1}.png')
        plt.savefig(output_path)
        plt.close()


generate_bitcoin_prices()
