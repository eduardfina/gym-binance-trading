import os

import numpy as np
import pandas as pd
import gymnasium as gym
import binance_trading_environment

from stable_baselines3 import PPO
from stable_baselines3.ppo import MultiInputPolicy
from stable_baselines3.common.monitor import Monitor


def ppo_train(name, timeframe, start_date, end_date, total_periods=1000000, investment=100000, verbose=0):
    env = gym.make(f'binance-v0-{timeframe}', initial_usdt=investment, start_date=start_date, end_date=end_date)

    log_dir = './logs/'
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir)
    model = PPO(MultiInputPolicy, env, verbose=verbose, tensorboard_log=log_dir)

    if verbose > 0:
        print("Train initialized: PPO_{}".format(name))

    model.learn(total_timesteps=total_periods, progress_bar=True)

    if verbose > 0:
        print("Train finished: PPO_{}".format(name))

    model.save('Models/PPO' + name + '_data.model')
    return model
