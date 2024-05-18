import pickle
import sys


def main():
    name = sys.argv[1]

    with open(f'results/A2C/A2C_{name}_pickle.pickle', 'rb') as f:
        data = pickle.load(f)
    print(f"Test results: {data}")


if __name__ == "__main__":
    main()
