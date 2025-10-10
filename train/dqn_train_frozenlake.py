# train_frozenlake.py
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
import os
import time
import pandas as pd  

ENV_NAME = "FrozenLake-v1"
IS_SLIPPERY = True
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
MODEL_PATH = "models\dqn_frozenlake.pth"

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

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

def one_hot(state, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[state] = 1.0
    return vec

def select_action(state, policy_net, epsilon, action_dim):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            q_values = policy_net(state)
            return q_values.argmax().item()

def train():
    start = time.time()
    env = gym.make(ENV_NAME, is_slippery=IS_SLIPPERY)
    state_space = env.observation_space.n
    action_space = env.action_space.n

    policy_net = DQN(state_space, action_space)
    target_net = DQN(state_space, action_space)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    epsilon = EPSILON_START
    rewards_history = []

    for episode in range(EPISODES):
        state, _ = env.reset()
        state = one_hot(state, state_space)
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, epsilon, action_space)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_encoded = one_hot(next_state, state_space)
            memory.add((state, action, reward, next_state_encoded, done))
            state = next_state_encoded
            total_reward += reward

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

        rewards_history.append(total_reward)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Epizoda {episode}, Nagrada: {total_reward}, Epsilon: {epsilon:.3f}")

    torch.save(policy_net.state_dict(), MODEL_PATH)
    env.close()

    end = time.time()
    print(f"Trening završen. Trajanje: {end - start:.2f} sekundi")

    # Analiza i graf
    rewards_series = pd.Series(rewards_history)
    window = 50
    rolling_mean = rewards_series.rolling(window).mean()
    rolling_std = rewards_series.rolling(window).std()

    plt.figure(figsize=(12, 6))
    plt.title("Deep Q-learning na FrozenLake-v1")
    plt.xlabel("Epizoda")
    plt.ylabel("Ukupna nagrada")

    plt.plot(rewards_history, color='lightgray', label="Originalna nagrada", alpha=0.3)
    plt.plot(rolling_mean, color='green', label=f"Prosječna nagrada ({window} epizoda)")
    plt.fill_between(range(len(rewards_history)),
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     color='green', alpha=0.2, label="Standardna devijacija")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("logs/dqn_frozenlake_learning_curve.png")
    plt.close()

if __name__ == "__main__":
    train()
