import sys

import gymnasium
import A2C
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
        name = f'A2C5mWindowSizePeriod{t}'

        if not TRAINED[t]:
            for p in periods:
                p = int(p)
                if p != t:
                    for num in range(4 - time):
                        if not created:
                            print(f"Training Model {name} with period {p} time {num+1}/{4- time}")
                            A2C.a2c_train(name, '5m', PERIODS[p][0], PERIODS[p][1])
                            created = True
                        else:
                            print(f"Continue training Model {name} with period {p} time {num+1}/{4- time}")
                            a2c_model = A2C.load_a2c_model(name)
                            A2C.a2c_continue_training(name, a2c_model, '5m', PERIODS[p][0], PERIODS[p][1])
                    time = 0

        if not TESTED[t]:
            print(f"Testing model {name} with period {t}")
            a2c_model = A2C.load_a2c_model(name)
            A2C.a2c_test(a2c_model, '5m', PERIODS[t][0], PERIODS[t][1], name)

            with open(f'results/A2C/A2C_{name}_pickle.pickle', 'rb') as f:
                data = pickle.load(f)
            print(f"Test results: {data}")


if __name__ == "__main__":
    main()
