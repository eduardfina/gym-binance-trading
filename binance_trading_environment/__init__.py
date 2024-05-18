from gymnasium.envs.registration import register
from . import datasets as ds

financial_data = ds.load_dataset('financial_data')
indicators_data = ds.load_dataset('indicators_data')

crypto_tickets = ['BTC', 'ETH', 'XRP', 'MATIC', 'ADA', 'BNB', 'LTC', 'LINK', 'DOGE', 'DOT', 'SHIB', 'SOL']
# crypto_tickets = ['BTC', 'ETH']


def load_crypto_data(timeframe):
    df = ds.load_dataset(f'crypto_data_{timeframe}')

    filtered_columns = [col for col in df.columns if any(name in col for name in crypto_tickets)]
    filtered_columns.append('timestamp')

    return df[filtered_columns]


register(
    id='binance-v0-5m',
    entry_point='binance_trading_environment.envs:TradingEnv',
    kwargs={
        'df': ds.load_dataset('crypto_data_5m'),
        'financial_df': financial_data,
        'indicators_df': indicators_data,
        'volume_trade': 0.1,
    }
)
'''
register(
    id='binance-v0-1h',
    entry_point='binance_trading_environment.envs:TradingEnv',
    kwargs={
        'df': load_crypto_data('1h'),
        'financial_df': financial_data,
        'volume_trade': 0.1,
    }
) '''

register(
    id='binance-v0-1d',
    entry_point='binance_trading_environment.envs:TradingEnv',
    kwargs={
        'df': ds.load_dataset('crypto_data_1d'),
        'financial_df': financial_data,
        'indicators_df': indicators_data,
        'volume_trade': 0.1,
    }
)
