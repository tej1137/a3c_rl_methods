#Dependencies
import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import numpy as np
import os

#enviroments and configuration
envs = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
algorithms = {"A2C" : A2C, "DQN" : DQN, "PPO" : PPO}
timesteps = 15000
eval_episodes = 10

results = {}

#Create new result folders
os.makedirs("results", exist_ok=True)

#Training env
for env_name in envs:
    print(f"\environment {env_name}")
    results[env_name] = {}

    env = gym.make(env_name)

    for algo_name, algo_class in algorithms.items():
        print(f" Training {algo_name} in {env_name}")

        model = algo_class("MlpPolicy", env, verbose=0)
        model.learn(total_timesteps=timesteps)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
        results[env_name][algo_name] = mean_reward
        print(f" {algo_name} Mean Reward: {mean_reward}")

    env.close()
#Save results
with open("results/results.txt", "w") as f:
    for env_name in results:
        f.write(f"environment: {env_name}")
        for algo in results[env_name]:
            f.write(f"{algo}: {results[env_name][algo]}")

#plot save & viz           
for env_name in results:
    algos = list(results[env_name].keys())
    scores = list(results[env_name].values())

    plt.figure()
    plt.bar(algos, scores)
    plt.ylabel("Mean Reward")
    plt.title(f"RL algorithms comparson on {env_name}")
    #savepng
    plt.savefig(f"results/{env_name}_comparispn.png")
    plt.close()



print("Results & figures saved")
