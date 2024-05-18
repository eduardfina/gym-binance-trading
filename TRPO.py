import os

import gymnasium as gym
import time
import pickle
import binance_trading_environment

from sb3_contrib import TRPO
from stable_baselines3.common.monitor import Monitor


def load_trpo_model(name):
    return TRPO.load('./Models/TRPO/TRPO_' + name + '_data.model', custom_objects={'lr_schedule': None})


def trpo_train(name, timeframe, start_date, end_date, total_periods=250000, investment=100000, verbose=0):
    env = gym.make(f'binance-v0-{timeframe}', initial_usdt=investment, start_date=start_date, end_date=end_date)

    log_dir = './logs/TRPO/'
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir)
    model = TRPO('MultiInputPolicy', env, verbose=verbose, tensorboard_log=log_dir, device='cuda')

    if verbose > 0:
        print("Train initialized: TRPO_{}".format(name))

    model.learn(total_timesteps=total_periods, progress_bar=True)

    if verbose > 0:
        print("Train finished: TRPO_{}".format(name))

    model.save('Models/TRPO/TRPO_' + name + '_data.model')
    return model


def trpo_continue_training(name, model, timeframe, start_date, end_date, total_periods=250000, investment=100000, verbose=0):
    env = gym.make(f'binance-v0-{timeframe}', initial_usdt=investment, start_date=start_date, end_date=end_date)

    log_dir = './logs/TRPO/'
    os.makedirs(log_dir, exist_ok=True)

    env = Monitor(env, log_dir)
    model.set_env(env)

    if verbose > 0:
        print("Train initialized: TRPO_{}".format(name))

    model.learn(total_timesteps=total_periods, progress_bar=True)

    if verbose > 0:
        print("Train finished: TRPO_{}".format(name))

    model.save('Models/TRPO/TRPO_' + name + '_data.model')
    return model


def trpo_test(model, timeframe, start_date, end_date, results_file, n_episodes=10, investment=100000):
    env = gym.make(f'binance-v0-{timeframe}', initial_usdt=investment, start_date=start_date, end_date=end_date)

    l_exe_time = []
    l_total_reward = []
    l_n_steps = []
    l_info = []

    info = {}

    for i in range(n_episodes):
        obs, _ = env.reset()

        total_reward = 0
        done = False
        truncated = False
        step = 0

        start_time = time.time()
        while not done and not truncated:
            actions, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(actions)
            total_reward += reward
            step += 1

        exe_time = (time.time() - start_time)/60
        l_exe_time.append(exe_time)
        l_total_reward.append(total_reward)
        l_n_steps.append(step)
        l_info.append(info)

        print(f'Episode {i+1} done!')

    print(f'\nTotal exe time: {l_exe_time}')
    print(f'\nTotal reward: {l_total_reward}')
    print(f'\nTotal steps: {l_n_steps}')
    print(f'\nFinal info: {l_info}')

    agent_trpo_pickle = {
        'l_total_reward': l_total_reward,
        'l_exe_time': l_exe_time,
        'l_n_steps': l_n_steps,
        'l_info': l_info
    }

    pickle.dump(agent_trpo_pickle, open('./results/TRPO/TRPO_' + results_file + '_pickle.pickle', 'wb'))

