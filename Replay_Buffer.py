import torch
import numpy as np
import gymnasium as gym

class ReplayBuffer():
    def __init__(self, observation_space_dim, action_space_dim, goal_space_dim, buffer_size):
        self.observation_space_dim = observation_space_dim
        self.action_space_dim = action_space_dim
        self.goal_space_dim = goal_space_dim
        self.buffer_size = buffer_size
        self.states = np.zeros((buffer_size, *observation_space_dim))
        self.actions = np.zeros((buffer_size, *action_space_dim))
        self.rewards = np.zeros(buffer_size)
        self.next_states = np.zeros((buffer_size, *observation_space_dim))
        self.dones = np.zeros(buffer_size)
        self.goals = np.zeros((buffer_size, *goal_space_dim))
        self.idx, self.filled = 0, 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    def add(self, obs, action, reward, next_obs, done, goal):
        self.states[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_states[self.idx] = next_obs
        self.dones[self.idx] = done
        self.goals[self.idx] = goal
        self.idx = (self.idx + 1) % self.buffer_size
        self.filled += 1

    def sample(self, batch_size):
        idxs = np.random.choice(min(self.filled, self.buffer_size),min(batch_size,self.filled), replace=False)
        return (torch.tensor(self.states[idxs], dtype=float).to(self.device), torch.tensor(self.actions[idxs], dtype=float).to(self.device),
                torch.tensor(self.rewards[idxs], dtype=float).to(self.device), torch.tensor(self.next_states[idxs], dtype=float).to(self.device),
                torch.tensor(self.goals[idxs], dtype=float).to(self.device), torch.tensor(self.dones[idxs], dtype=float).to(self.device))