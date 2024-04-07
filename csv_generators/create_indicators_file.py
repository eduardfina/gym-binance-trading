import pandas as pd
from datetime import datetime

HALVING_DATES = ['2020-05-11', '2024-04-19', '2028-03-27']


def create_indicators_file():
    start_date = datetime.strptime('2017-08-17', '%Y-%m-%d')
    end_date = datetime.strptime('2024-04-01', '%Y-%m-%d')
    dates = pd.date_range(start=start_date, end=end_date, freq='D')

    crypto_sentiment = pd.read_csv('indicators/crypto_sentiment.csv', parse_dates=['timestamp'])
    crypto_sentiment['timestamp'] = crypto_sentiment['timestamp'].dt.tz_localize(None)
    crypto_sentiment = crypto_sentiment[['timestamp', 'value']]

    result = pd.DataFrame({'Dates': dates})
    result['Dates'] = result['Dates'].dt.tz_localize(None)

    result = pd.merge(result, crypto_sentiment, left_on='Dates', right_on='timestamp', how='left')
    result['value'] = result['value'].fillna(50)

    result['Days_until_halving'] = result['Dates'].apply(lambda x: min((datetime.strptime(date, '%Y-%m-%d') - x).days for date in HALVING_DATES if datetime.strptime(date, '%Y-%m-%d') > x))
    result.drop(columns=['timestamp'], inplace=True)

    result = result[['Dates', 'value', 'Days_until_halving']].rename(columns={'value': 'Sentiment'})
    result['Sentiment'] = result['Sentiment'].astype(int)

    # Save to CSV
    result.to_csv('../binance_trading_environment/datasets/data/indicators_data.csv', index=False)


create_indicators_file()
