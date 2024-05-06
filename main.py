import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Assuming `model` is your neural network model for DQN
class DQNAgent:
    def __init__(
        self,
        model,
        replay_buffer_size=10000,
        batch_size=64,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
    ):
        self.model = model
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.optimizer = optim.Adam(self.model.parameters())
        self.criterion = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(
                0, self.model.output_dim
            )  # Explore: select a random action
        else:
            with torch.no_grad():
                return (
                    self.model(state).argmax(dim=1).item()
                )  # Exploit: select the action with the highest Q-value

    def replay(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * self.model(next_state).max().item()
            target_f = self.model(state)
            target_f[action] = target
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state))
            loss.backward()
            self.optimizer.step()

    def train(self, env, episodes):
        for episode in range(episodes):
            state = env.reset()
            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if len(self.replay_buffer) > self.batch_size:
                    self.replay(self.batch_size)
            self.epsilon = max(self.epsilon_end, self.epsilon_decay * self.epsilon)


# Example usage
env = gym.make("CartPole-v0")
agent = DQNAgent(model)
agent.train(env, episodes=100)
