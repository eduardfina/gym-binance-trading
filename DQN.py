import os
import pickle

import numpy as np
import gymnasium as gym
import binance_trading_environment
from datetime import datetime

from DQN_torch import DQN


def load_dqn_model(model, name):
    return model.load('./Models/DQN/DQN_' + name + '_data.model')


def dqn_train(name, timeframe, start_date, end_date, total_periods=1000, investment=100000, verbose=0):
    env = gym.make(f'binance-v0-{timeframe}', initial_usdt=investment, start_date=start_date, end_date=end_date)

    state_size, action_size = env.get_sizes()

    model = DQN(env, state_size, action_size)

    if verbose > 0:
        print("Train initialized: DQN_{}".format(name))

    model.train(n_episodes=total_periods, verbose=verbose)

    if verbose > 0:
        print("Train finished: DQN_{}".format(name))

    model.save_model('Models/DQN/DQN_' + name + '_data')
    model.save_stats(f'logs/DQN/Model{name}')
    return model


def dqn_continue_training(name, timeframe, start_date, end_date, total_periods=1000, investment=100000, verbose=0):
    env = gym.make(f'binance-v0-{timeframe}', initial_usdt=investment, start_date=start_date, end_date=end_date)

    state_size, action_size = env.get_sizes()

    model = DQN(env, state_size, action_size)
    model.load_model('Models/DQN/DQN_' + name + '_data')

    if verbose > 0:
        print("Train initialized: DQN_{}".format(name))

    model.train(n_episodes=total_periods, verbose=verbose)

    if verbose > 0:
        print("Train finished: DQN_{}".format(name))

    model.save_model('Models/DQN/DQN_' + name + '_data')
    model.save_stats(f'logs/DQN/Model{name}')
    return model


def dqn_test(name, timeframe, start_date, end_date, investment=100000):
    env = gym.make(f'binance-v0-{timeframe}', initial_usdt=investment, start_date=start_date, end_date=end_date)

    state_size, action_size = env.get_sizes()

    model = DQN(env, state_size, action_size)

    l_rewards = []
    l_info = []

    for i in range(1, 5):
        model.load_model('Models/DQN/DQN_' + name + '_data')
        test_reward, test_info = model.test()
        l_rewards.append(test_reward)
        l_info.append(test_info)
        print(test_info[-1])
        print(f"Episode reward {i} = {test_reward}")

    print("Mean total rewards {:2f}, variance {:2f}".format(np.mean(l_rewards), np.std(l_rewards)))

    data = {'l_total_reward': l_rewards,
            'info': l_info}

    now = datetime.now()
    pickle.dump(data, open('./results/DQN/DQN_' + name + '¨_data_' + now.strftime("%d-%b_%H_%M_%S") + '.pickle', 'wb'))
