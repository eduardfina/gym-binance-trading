import pandas as pd
import glob

TIMES = ['5m', '1h', '1d']


def combine_crypto_data():
    for TIME in TIMES:
        print(f'Processing {TIME} data')
        # Get all csv files with pattern 'CRYPTO_USDT_{TIME}.csv'
        files = glob.glob(f"crypto/*_USDT-{TIME}.csv")

        # Create an empty DataFrame to store combined data
        combined_data = pd.DataFrame()

        print('Adding BTC values')
        BTC_data = pd.read_csv(f'crypto/BTC_USDT-{TIME}.csv')
        BTC_data = BTC_data.rename(columns={'open': 'BTC_open', 'high': 'BTC_high', 'low': 'BTC_low', 'close': 'BTC_close', 'volume': 'BTC_volume'})
        combined_data = BTC_data

        for file in files:
            if file == f'crypto\\BTC_USDT-{TIME}.csv':
                continue

            # Read each csv file
            data = pd.read_csv(file)

            # Extract cryptocurrency ticker from the filename
            crypto_ticker = file.split('\\')[1].split('_')[0]
            print(f"Adding {crypto_ticker} values")

            # Select only timestamp and close columns and rename the close column to the crypto ticker
            data = data.rename(columns={'open': crypto_ticker+'_open', 'high': crypto_ticker+'_high', 'low': crypto_ticker+'_low', 'close': crypto_ticker+'_close', 'volume': crypto_ticker+'_volume'})

            if combined_data.empty:
                combined_data = data
            else:
                # Merge DataFrames on timestamp column
                combined_data = pd.merge(combined_data, data, on='timestamp', how='outer')

        # Fill the na values with 0
        combined_data = combined_data.fillna(0)
        # Save the combined DataFrame to a new csv file
        combined_data.to_csv(f'../binance_trading_environment/datasets/data/crypto_data_{TIME}.csv', index=False)


combine_crypto_data()
