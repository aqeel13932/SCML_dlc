import numpy as np


class ReplayMemory():
    def __init__(self, size, observation_shape):
        self.size = size
        self.count = 0
        self.current = 0

        self.observations = np.empty((size,) + observation_shape, dtype=np.float32)
        self.actions = np.empty(size, dtype=np.uint8)
        self.rewards = np.empty(size, dtype=np.float32)
        self.dones = np.empty(size, dtype=np.bool)
        self.next_observations = np.empty((size,) + observation_shape, dtype=np.float32)

    def add(self, obs, action, reward, done, next_obs):
        self.observations[self.current] = obs
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.dones[self.current] = done
        self.next_observations[self.current] = next_obs

        self.current += 1
        self.count = max(self.current, self.count)
        self.current %= self.size

    def sample(self, batch_size):
        idx = np.random.choice(self.count, batch_size, replace=False)

        obs = self.observations[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]
        next_obs = self.next_observations[idx]

        return obs, actions, rewards, dones, next_obs
