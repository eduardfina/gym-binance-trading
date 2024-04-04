from time import time
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


class Actions(Enum):
    Hold = 0
    Sell = 1
    Buy = 2


class TradingEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 3}

    def __init__(self, df, financial_df, initial_usdt, start_date, end_date, volume_trade=0.05, render_mode=None):

        self.render_mode = render_mode

        self.df = df
        self.financial_df = financial_df
        self.last_index = df.index[-1]

        self.initial_usdt = initial_usdt
        self.volume_trade = volume_trade

        self.crypto_tickets = [col[:-6] for col in self.df if col.endswith('_close')]
        self.portfolio = [0 for _ in range(len(self.crypto_tickets))]
        self.trading_fee = 0.01

        self.max_loss = 0.2
        self.actual_profit = initial_usdt
        self.total_reward = 0

        self.total_trading_fee = 0
        self.initial_period = df.index[df['timestamp'].str.contains(start_date)].values[0]
        self.end_period = df.index[df['timestamp'].str.contains(end_date)].values[0]

        self.action_space = spaces.MultiDiscrete([len(Actions) for _ in range(len(self.portfolio))])
        self.observation_space = spaces.Dict({
            'usdt': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'crypto_data': spaces.Box(low=0, high=np.inf, shape=(self.df.shape[1]-1,)),
            'financial_data': spaces.Box(low=-np.inf, high=np.inf, shape=(self.financial_df.shape[1]-1,)),
            'portfolio': spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(len(self.portfolio),), dtype=np.int64)
        })

        self.usdt = initial_usdt
        self.period = self.initial_period

    def _get_obs(self):
        date = self.df.loc[self.period, 'timestamp'].split(' ')[0]
        date_index = self.financial_df.loc[self.financial_df['Dates'] == date].index[0]

        obs = {}

        try:
            obs = {
                'usdt': np.array(self.usdt, dtype=np.float32).reshape(1, ),
                'crypto_data': self.df.loc[self.period, self.df.columns != 'timestamp'].to_numpy(dtype=np.float32),
                'financial_data': self.financial_df.loc[date_index, self.financial_df.columns != 'Dates'].to_numpy(dtype=np.float32),
                'portfolio': np.array(self.portfolio, dtype=np.int64)
            }
        except:
            print('Fuck!')

        return obs

    def reset(self, seed=None, options=None):

        self.portfolio = [0 for _ in range(len(self.crypto_tickets))]
        self.total_trading_fee = 0

        self.action_space = spaces.MultiDiscrete([len(Actions) for _ in range(len(self.portfolio))])
        self.observation_space = spaces.Dict({
            'usdt': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'crypto_data': spaces.Box(low=0, high=np.inf, shape=(self.df.shape[1]-1,)),
            'financial_data': spaces.Box(low=-np.inf, high=np.inf, shape=(self.financial_df.shape[1]-1,)),
            'portfolio': spaces.Box(low=0, high=np.iinfo(np.int64).max, shape=(len(self.portfolio),), dtype=np.int64)
        })

        self.usdt = self.initial_usdt
        self.actual_profit = self.initial_usdt
        self.period = self.initial_period

        return self._get_obs(), self._get_info()

    def _sell_cryptos(self, actions):

        actions_sell = np.where(np.array(actions) == 1)[0]

        for i in actions_sell:
            if self.portfolio[i] > 0:
                crypto_price = self.df.at[self.period, str(self.crypto_tickets[i]) + '_close']
                if crypto_price > 0:
                    usdt_value = np.round(self.portfolio[i] * crypto_price, 2)
                    sell_fee = np.round(usdt_value * self.trading_fee, 2)
                    self.usdt += usdt_value - sell_fee
                    self.portfolio[i] = 0

    def _buy_cryptos(self, actions):

        actions_buy = np.where(np.array(actions) == 2)[0]
        usdt_per_invest = np.round(self.usdt * self.volume_trade)
        buy_fee = np.round(usdt_per_invest * self.trading_fee, 2)

        for i in np.random.choice(actions_buy, len(actions_buy), replace=False):
            if self.usdt >= usdt_per_invest:
                crypto_price = self.df.at[self.period, str(self.crypto_tickets[i]) + '_close']
                if crypto_price > 0:
                    self.usdt -= usdt_per_invest
                    self.portfolio[i] += np.round((usdt_per_invest - buy_fee) / crypto_price, 2)

    def _get_actual_profits(self):
        prices = self.df.loc[self.period, self.df.columns.str.endswith('_close')]

        portfolio_value = 0

        for index, price in enumerate(prices):
            if isinstance(price, str) or isinstance(self.portfolio[index], str):
                print(price)

            portfolio_value += price * self.portfolio[index]

        return np.round(portfolio_value + self.usdt, 2)

    def _get_info(self):
        return dict(
            usdt=self.usdt,
            actual_profit=self.actual_profit,
            portfolio=self.portfolio
        )

    def step(self, actions):

        self._sell_cryptos(actions)
        self._buy_cryptos(actions)

        new_profit = self._get_actual_profits()

        reward = new_profit - self.actual_profit
        self.actual_profit = new_profit

        state = self._get_obs()
        info = self._get_info()

        if self.period >= self.end_period or (self.actual_profit < self.initial_usdt * self.max_loss):
            if self.period == self.end_period:
                print(f'Hell yeah! Profits: {self.actual_profit}')
            truncated = True
        else:
            truncated = False

        self.period += 1

        return state, reward, False, truncated, info
