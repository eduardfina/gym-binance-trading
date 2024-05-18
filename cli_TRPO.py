import sys

import gymnasium
import TRPO
import pickle

# Divide the data in 5 equal periods
PERIODS = [
    ['2017-08-27', '2019-01-28'],
    ['2019-01-29', '2020-07-10'],
    ['2020-07-11', '2021-12-21'],
    ['2021-12-22', '2023-06-03'],
    ['2023-06-04', '2024-03-05']
]

TRAINED = [False, False, False, False, False]
TESTED = [False, False, False, False, False]


def main():
    arr = sys.argv[1].split(',')
    periods = sys.argv[2].split(',')
    created = (sys.argv[3] == 'True')
    time = int(sys.argv[4])

    for t in arr:
        t = int(t)
        name = f'1dPeriod{t}'

        if not TRAINED[t]:
            for p in periods:
                p = int(p)
                if p != t:
                    for num in range(4 - time):
                        if not created:
                            print(f"Training Model {name} with period {p} time {num+1}/{4- time}")
                            TRPO.trpo_train(name, '1d', PERIODS[p][0], PERIODS[p][1])
                            created = True
                        else:
                            print(f"Continue training Model {name} with period {p} time {num+1}/{4- time}")
                            trpo_model = TRPO.load_trpo_model(name)
                            TRPO.trpo_continue_training(name, trpo_model, '1d', PERIODS[p][0], PERIODS[p][1])
                    time = 0

        if not TESTED[t]:
            print(f"Testing model {name} with period {t}")
            trpo_model = TRPO.load_trpo_model(name)
            TRPO.trpo_test(trpo_model, '1d', PERIODS[t][0], PERIODS[t][1], name)

            with open(f'results/TRPO/TRPO_{name}_pickle.pickle', 'rb') as f:
                data = pickle.load(f)
            print(f"Test results: {data}")


if __name__ == "__main__":
    main()
