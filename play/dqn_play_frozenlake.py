import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import os
import time

ENV_NAME = "FrozenLake-v1"
IS_SLIPPERY = True
MODEL_PATH = "dqn_frozenlake.pth"

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

def one_hot(state, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[state] = 1.0
    return vec

def select_action(state, policy_net):
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = policy_net(state)
        return q_values.argmax().item()

def evaluate(episodes=100, visualize_first_n=2):
    if not os.path.exists(MODEL_PATH):
        print("Model nije pronađen.")
        return

    # Okoliši
    env_render = gym.make(ENV_NAME, is_slippery=IS_SLIPPERY, render_mode="human")
    env_eval = gym.make(ENV_NAME, is_slippery=IS_SLIPPERY)

    state_space = env_eval.observation_space.n
    action_space = env_eval.action_space.n

    # Model
    policy_net = DQN(state_space, action_space)
    policy_net.load_state_dict(torch.load(MODEL_PATH))
    policy_net.eval()

    rewards = []

    for episode in range(episodes):
        env = env_render if episode < visualize_first_n else env_eval
        state, _ = env.reset()
        state = one_hot(state, state_space)

        total_reward = 0
        done = False
        steps = 0

        while not done:
            if episode < visualize_first_n:
                time.sleep(0.3)  # Sporije za vizualni prikaz

            action = select_action(state, policy_net)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = one_hot(next_state, state_space)
            total_reward += reward
            steps += 1

        rewards.append(total_reward)

        if episode < visualize_first_n:
            print(f"[Vizualizacija] Epizoda {episode}, Nagrada: {total_reward}, Koraka: {steps}")

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print("\n--- Evaluacija modela DQN na FrozenLake-v1 ---")
    print(f"Mean reward over {episodes} episodes: {mean_reward:.2f} +/- {std_reward:.2f}")

    env_render.close()
    env_eval.close()

if __name__ == "__main__":
    evaluate()
