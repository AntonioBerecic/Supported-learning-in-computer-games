# train_dqn.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import time
import os
import pandas as pd

# Postavke
ENV_NAME = "CartPole-v1"
GAMMA = 0.99
LR = 1e-3
BATCH_SIZE = 64
MEMORY_SIZE = 10000
MIN_MEMORY_SIZE = 1000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE_FREQ = 10
EPISODES = 5000
MODEL_PATH = "models\dqn_cartpole.pth"

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

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(np.array(state)).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()

def train():
    start = time.time()
    env = gym.make(ENV_NAME)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(obs_dim, action_dim)
    target_net = DQN(obs_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    rewards_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, epsilon, action_dim)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.add((state, action, reward, np.array(next_state), done))
            state = next_state
            episode_reward += reward

            if len(memory) >= MIN_MEMORY_SIZE:
                batch = memory.sample(BATCH_SIZE)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(np.array(states))
                actions = torch.LongTensor(actions).unsqueeze(1)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(np.array(next_states))
                dones = torch.FloatTensor(dones).unsqueeze(1)

                current_q = policy_net(states).gather(1, actions)
                next_q = target_net(next_states).max(1)[0].unsqueeze(1)
                target_q = rewards + GAMMA * next_q * (1 - dones)

                loss = nn.MSELoss()(current_q, target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rewards_history.append(episode_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Epizoda {episode}, Nagrada: {episode_reward}, Epsilon: {epsilon:.3f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    env.close()

    end = time.time()
    duration = end - start
    print(f"Trening završen. Trajanje: {duration:.2f} sekundi ({duration/60:.2f} minuta)")

    # Analiza rezultata
    rewards_series = pd.Series(rewards_history)
    window = 50 
    rolling_mean = rewards_series.rolling(window).mean()
    rolling_std = rewards_series.rolling(window).std()

    # Crtanje grafa
    plt.figure(figsize=(12, 6))
    plt.title("Deep Q-learning na CartPole-v1")
    plt.xlabel("Epizoda")
    plt.ylabel("Ukupna nagrada")

    plt.plot(rewards_history, color='lightgray', label="Originalna nagrada", alpha=0.3)
    plt.plot(rolling_mean, color='blue', label=f"Prosječna nagrada ({window} epizoda)")
    plt.fill_between(range(len(rewards_history)),
                    rolling_mean - rolling_std,
                    rolling_mean + rolling_std,
                    color='blue', alpha=0.2, label="Standardna devijacija")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("logs/dqn_cartpole_learning_curve.png")
    plt.close()



if __name__ == "__main__":
    train()
