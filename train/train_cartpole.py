import gymnasium as gym
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
import json
import time

def format_time(minutes):
    total_seconds = int(minutes * 60)
    mins = total_seconds // 60
    secs = total_seconds % 60
    return f"{mins} minuta i {secs} sekundi"

# Definiranje custom callbacka za praćenje performansi tijekom treninga
class TrainAndEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, n_eval_episodes=10, verbose=1):
        super(TrainAndEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations_mean_reward = []
        self.evaluations_std_reward = []
        self.timesteps = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward, std_reward = evaluate_policy(self.model, self.eval_env,
                                                      n_eval_episodes=self.n_eval_episodes,
                                                      render=False)
            self.evaluations_mean_reward.append(mean_reward)
            self.evaluations_std_reward.append(std_reward)
            self.timesteps.append(self.num_timesteps)

            if self.verbose > 0:
                print(f"Num timesteps: {self.num_timesteps}, Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        return True

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = f"./logs/cartpole_policy_gradient_logs_{timestamp}/"
os.makedirs(LOG_DIR, exist_ok=True)

TOTAL_TIMESTEPS = 20000
EVAL_FREQ = 500
N_EVAL_EPISODES = 10

# Rjecnik za pohranu callback podataka i vremena treniranja
performance_data = {}

print(f"Trening CartPole-v1 agenata će se spremati u direktorij: {LOG_DIR}")

## 1. Treniranje A2C agenta

print("\n--- Pokrećem trening A2C agenta ---")
env_a2c = gym.make("CartPole-v1")
eval_env_a2c = Monitor(gym.make("CartPole-v1"))

model_a2c = A2C("MlpPolicy", env_a2c, verbose=0, tensorboard_log=LOG_DIR)
callback_a2c = TrainAndEvalCallback(eval_env_a2c, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)

start_time_a2c = time.time()
model_a2c.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_a2c)
end_time_a2c = time.time()
training_time_a2c_minutes = (end_time_a2c - start_time_a2c) / 60

model_a2c.save(os.path.join(LOG_DIR, "a2c_cartpole"))
env_a2c.close()
eval_env_a2c.close()
performance_data['A2C'] = {'timesteps': callback_a2c.timesteps,
                            'mean_reward': callback_a2c.evaluations_mean_reward,
                            'std_reward': callback_a2c.evaluations_std_reward,
                            'training_time_minutes': training_time_a2c_minutes}
print(f"A2C model spremljen u: {os.path.join(LOG_DIR, 'a2c_cartpole.zip')}")
print(f"A2C trening vrijeme: {format_time(training_time_a2c_minutes)}")

## 2. Treniranje PPO agenta

print("\n--- Pokrećem trening PPO agenta ---")
env_ppo = gym.make("CartPole-v1")
eval_env_ppo = Monitor(gym.make("CartPole-v1"))

model_ppo = PPO("MlpPolicy", env_ppo, verbose=0, tensorboard_log=LOG_DIR)
callback_ppo = TrainAndEvalCallback(eval_env_ppo, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)

start_time_ppo = time.time()
model_ppo.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback_ppo)
end_time_ppo = time.time()
training_time_ppo_minutes = (end_time_ppo - start_time_ppo) / 60

model_ppo.save(os.path.join(LOG_DIR, "ppo_cartpole"))
env_ppo.close()
eval_env_ppo.close()
performance_data['PPO'] = {'timesteps': callback_ppo.timesteps,
                            'mean_reward': callback_ppo.evaluations_mean_reward,
                            'std_reward': callback_ppo.evaluations_std_reward,
                            'training_time_minutes': training_time_ppo_minutes}
print(f"PPO model spremljen u: {os.path.join(LOG_DIR, 'ppo_cartpole.zip')}")
print(f"PPO trening vrijeme: {format_time(training_time_ppo_minutes)}")

## Spremanje podataka o performansama

with open(os.path.join(LOG_DIR, 'performance_data.json'), 'w') as f:
    serializable_performance_data = {
        algo: {
            'timesteps': data['timesteps'],
            'mean_reward': [float(x) for x in data['mean_reward']],
            'std_reward': [float(x) for x in data['std_reward']],
            'training_time_minutes': float(data['training_time_minutes'])
        }
        for algo, data in performance_data.items()
    }
    json.dump(serializable_performance_data, f, indent=4)
print(f"Podaci o performansama spremljeni u: {os.path.join(LOG_DIR, 'performance_data.json')}")

print("\n--- Trening CartPole-v1 agenata završen. ---")
print(f"Spremljeni modeli i podaci nalaze se u direktoriju: {LOG_DIR}")
print("Sada možete koristiti 'play_cartpole.py' za vizualizaciju i finalnu evaluaciju.")