#Dependencies
import gymnasium as gym
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import os
import time
#enviroments and configuration
envs = ["CartPole-v1", "Taxi-v3", "Acrobot-v1"]
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

        #training time calcul. for each algo
        start_time = time.time()
        model.learn(total_timesteps=timesteps)
        end_time = time.time()
        train_time = end_time - start_time

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)

        results[env_name][algo_name] = mean_reward, train_time

        print(f" {algo_name} Mean Reward: {mean_reward}")
        print(f"{algo_name} Training time in sec : {train_time:.2f}")

    env.close()
#Save results
with open("results/results.txt", "w") as f:
    for env_name in results:
        f.write(f"environment: {env_name}")
        for algo in results[env_name]:

            value = results[env_name][algo]

            #update suporrt both tuple and dic formats
            if isinstance(value, dict):
                reward= value["reward"]
                time_taken = value["time"]
            else: #fallback tupleee!!!
                reward = value[0]
                time_taken = value[1]

            f.write(f"{algo}: Mean Reward :{reward:.2f} Training Time: {time_taken:.2f}\n")

#plot save & viz           
for env_name in results:
    algos = list(results[env_name].keys())
    scores = []
    for algo in algos: 
        value = results[env_name][algo]
        if isinstance(value, dict):
            scores.append(value["reward"])
        else:
            scores.append(value[0])

    plt.figure()
    plt.bar(algos, scores)
    plt.ylabel("Mean Reward")
    plt.title(f"RL algorithms comparson on {env_name}")
    #savepng
    plt.savefig(f"results/{env_name}_comparispn.png")
    plt.close()



print("Results & figures saved")
