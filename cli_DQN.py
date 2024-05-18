import gymnasium
import DQN
import pickle

# Divide the data in 5 equal periods
PERIODS = [
    ['2017-08-27', '2019-01-28'],
    ['2019-01-29', '2020-07-10'],
    ['2020-07-11', '2021-12-21'],
    ['2021-12-22', '2023-06-03'],
    ['2023-06-04', '2024-03-05']
]


def main():
    for t in range(len(PERIODS)):
        name = f'DQN5mWindowSizePeriod{t}'

        created = False
        for p in range(len(PERIODS)):
            if p != t:
                if not created:
                    print(f"Training Model {name} with period {p}")
                    DQN.dqn_train(name, '5m', PERIODS[p][0], PERIODS[p][1])
                    created = True
                else:
                    print(f"Continue training Model {name} with period {p}")
                    DQN.dqn_continue_training(name, '5m', PERIODS[p][0], PERIODS[p][1])

        print(f"Testing model {name} with period {t}")
        DQN.dqn_test(name, '5m', PERIODS[t][0], PERIODS[t][1])

        with open(f'results/DQN_{name}_pickle.pickle', 'rb') as f:
            data = pickle.load(f)
        print(f"Test results: {data}")


if __name__ == "__main__":
    main()
