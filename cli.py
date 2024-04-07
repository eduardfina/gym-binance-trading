import gymnasium
import PPO
import pickle


def main():
    name = 'firstModelPPO'

    PPO.ppo_train(name, '5m', '2017-08-17', '2022-07-01')
    print("Env loaded")

    ppo_model = PPO.load_ppo_model(name)

    PPO.ppo_test(ppo_model, '5m', '2022-07-01', '2024-03-05', name)

    with open('results/PPO_firstModelPPO_pickle.pickle', 'rb') as f:
        data = pickle.load(f)
        print(data)


if __name__ == "__main__":
    main()
