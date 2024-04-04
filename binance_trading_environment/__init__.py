from gymnasium.envs.registration import register
from . import datasets as ds

financial_data = ds.load_dataset('financial_data')

'''
register(
    id='binance-v0-5m',
    entry_point='binance_trading_environment.envs:TradingEnv',
    kwargs={
        'df': ds.load_dataset('crypto_data_5m'),
        'financial_df': financial_data,
        'volume_trade': 0.1,
    }
)
'''
register(
    id='binance-v0-1h',
    entry_point='binance_trading_environment.envs:TradingEnv',
    kwargs={
        'df': ds.load_dataset('crypto_data_1h'),
        'financial_df': financial_data,
        'volume_trade': 0.1,
    }
)
'''
register(
    id='binance-v0-1d',
    entry_point='binance_trading_environment.envs:TradingEnv',
    kwargs={
        'df': ds.load_dataset('crypto_data_1d'),
        'financial_df': financial_data,
        'volume_trade': 0.1,
    }
)
'''