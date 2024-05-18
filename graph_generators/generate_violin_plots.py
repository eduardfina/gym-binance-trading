import os
import pickle
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt

RESULTS_PATH = '../results/'
OUTPUT_DIR = 'images'

# Divide the data in 5 equal periods
PERIODS = [
    ['2017/08/27', '2019/01/28'],
    ['2019/01/29', '2020/07/10'],
    ['2020/07/11', '2021/12/21'],
    ['2021/12/22', '2023/06/03'],
    ['2023/06/04', '2024/03/05']
]


def generate_violin_plots():
    for r in ['5m', '1d']:
        for i in range(5):
            rewards_dict = {'PPO': [], 'A2C': [], 'TRPO': []}
            for m in ['PPO', 'A2C', 'TRPO']:
                with open(f'../results/{m}/{m}_{r}Period{i}_pickle.pickle', 'rb') as f:
                    data = pickle.load(f)
                    rewards_dict[m].extend(data['l_total_reward'])

            # Create DataFrame from dictionary
            df = pd.DataFrame(rewards_dict)

            # Generate violin plots
            plt.figure(figsize=(15, 10))
            sns.violinplot(data=df)
            plt.title(f'Comparison of Total Rewards by Algorithm ({PERIODS[i][0]} - {PERIODS[i][1]})')
            plt.xlabel('Algorithm')
            plt.ylabel('Total Reward')
            plt.xticks(rotation=45)
            plt.grid(True)

            # Save the plot as a PNG file
            plt.savefig(os.path.join(OUTPUT_DIR, f'comparison_profits_{r}_period{i}.png'))
            plt.close()


generate_violin_plots()
