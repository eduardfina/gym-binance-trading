import pickle
from datetime import datetime

import numpy as np

import binance_trading_environment
import gymnasium as gym
from DQN_torch import DQN

TIMEFRAME = '5m'
INVESTMENT = 100000
START_DATE = '2017-08-17'
END_DATE = '2022-07-01'

TRAIN = True
N_EPISODES = 1000
VERBOSE = 1


def main():
    env = gym.make(f'binance-v0-{TIMEFRAME}', initial_usdt=INVESTMENT, start_date=START_DATE, end_date=END_DATE)

    state_size, action_size = env.get_sizes()

    dqn_agent = DQN(env, state_size, action_size)
    # Entreno modelo DQN
    now = datetime.now()
    print("Entrenando modelo DQN_{}: {} ".format(TIMEFRAME, now.strftime("%d-%b %H:%M:%S")))
    dqn_agent.train(n_episodes=N_EPISODES, verbose=VERBOSE)

    dqn_agent.save_model(F'Models/DQN/Model{TIMEFRAME}')
    dqn_agent.save_stats(f'logs/DQN/Model{TIMEFRAME}')

    l_rewards = []
    l_info = []

    now = datetime.now()
    for i in range(1, 3):
        dqn_agent_test = DQN(env, state_size, action_size)
        dqn_agent_test.load_model(f'Models/DQN/Model{TIMEFRAME}')
        test_reward, test_info = dqn_agent_test.test()
        l_rewards.append(test_reward)
        l_info.append(test_info)
        if VERBOSE == 1:
            print(test_info[-1])
            print(f"Reward episodio {i} = {test_reward}")

    print("Media de total reward {:2f}, varianza {:2f}".format(np.mean(l_rewards), np.std(l_rewards)))

    agente_DQN_pickle = {'l_total_reward': l_rewards,
                         'info': l_info}

    pickle.dump(agente_DQN_pickle,open('./results/DQN_'+ TIMEFRAME + '-'+ now.strftime("%d-%b_%H_%M_%S") + '.pickle', 'wb'))


if __name__ == "__main__":
    main()
