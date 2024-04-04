import gymnasium
import PPO


def main():
    finalModel = PPO.ppo_train('firstModelPPO', '1h', '2020-01-01', '2022-01-01')
    print("Env loaded")


if __name__ == "__main__":
    main()
