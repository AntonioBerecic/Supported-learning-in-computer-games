import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
import time

ENV_NAME = "LunarLander-v3"
MODEL_PATH = "models\dqn_moonlander.pth"

class DQN(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

def select_action(state, policy_net, action_dim):
    with torch.no_grad():
        state = torch.FloatTensor(np.array(state)).unsqueeze(0)
        q_values = policy_net(state)
        return q_values.argmax().item()

def evaluate(episodes=100, visualize_first_n=2):
    if not os.path.exists(MODEL_PATH):
        print("Model nije pronađen. Pokreni najprije trening.")
        return

    # Prikaz za prve epizode
    env_render = gym.make(ENV_NAME, render_mode="human")
    # Ostale epizode bez prikaza
    env_eval = gym.make(ENV_NAME)

    obs_dim = env_eval.observation_space.shape[0]
    action_dim = env_eval.action_space.n

    policy_net = DQN(obs_dim, action_dim)
    policy_net.load_state_dict(torch.load(MODEL_PATH))
    policy_net.eval()

    rewards = []

    for episode in range(episodes):
        env = env_render if episode < visualize_first_n else env_eval

        state, _ = env.reset()
        episode_reward = 0
        done = False
        step_count = 0

        while not done:
            if episode < visualize_first_n:
                time.sleep(0.01)

            action = select_action(state, policy_net, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward
            step_count += 1

        rewards.append(episode_reward)

        if episode < visualize_first_n:
            print(f"[Vizualizacija] Epizoda {episode}, Nagrada: {episode_reward:.2f}, Koraka: {step_count}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print("\n--- Evaluacija modela DQN na LunarLander-v3 ---")
    print(f"Mean reward over {episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    env_render.close()
    env_eval.close()

if __name__ == "__main__":
    evaluate()
