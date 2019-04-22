import numpy as np
import random


class ReplayMemory():
    def __init__(self, size, observation_shape, hist_len):
        self.size = size
        self.count = 0
        self.current = 0
        self.hist_len = hist_len
        self.observation_shape = observation_shape

        self.observations = np.empty((size,) + observation_shape, dtype=np.uint8)
        self.actions = np.empty(size, dtype=np.uint8)
        self.rewards = np.empty(size, dtype=np.float32)
        self.dones = np.empty(size, dtype=np.bool)

    def add(self, obs, action, reward, done):
        self.observations[self.current] = obs
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        self.dones[self.current] = done

        self.current += 1
        self.count = max(self.current, self.count)
        self.current %= self.size

    def sample(self, batch_size):
        assert self.count - self.hist_len > batch_size

        n = 0
        idx = np.empty(batch_size, dtype=np.uint32)
        obs = np.empty((batch_size, self.observation_shape[0], self.observation_shape[1], self.observation_shape[2] * self.hist_len), dtype=np.uint8)
        next_obs = np.empty((batch_size, self.observation_shape[0], self.observation_shape[1], self.observation_shape[2] * self.hist_len), dtype=np.uint8)
        while n < batch_size:
            # reserve hist_len in the beginning of replay memory
            i = random.randint(self.hist_len, self.count - 1)
            # i is the index of _next_ observation
            # skip if episode ended within previous hist_len frames
            # or if current pointer is within those hist_len frames
            if np.any(self.dones[(i - self.hist_len):(i - 1)]) \
                    or (i - self.hist_len) < self.current <= i:
                #print("not good:", i)
                continue
            else:
                #print("good:", i)
                pass
            # take slice of last hist_len observations
            obs[n] = np.swapaxes(self.observations[(i - self.hist_len):i], 0, 3)[0]
            next_obs[n] = np.swapaxes(self.observations[(i - self.hist_len + 1):(i + 1)], 0, 3)[0]
            idx[n] = i - 1
            n += 1

        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]
        return obs, actions, rewards, dones, next_obs


if __name__ == '__main__':
    mem = ReplayMemory(10, (84, 84, 1), 4)
    assert mem.size == 10
    assert mem.current == 0
    assert mem.count == 0

    assert mem.observations.shape == (10, 84, 84, 1)
    assert mem.actions.shape == (10,)
    assert mem.rewards.shape == (10,)
    assert mem.dones.shape == (10,)

    for i in range(25):
        mem.add(np.ones((84, 84, 1)) * i, i, i, i % 17 == 0)
        assert mem.size == 10
        assert mem.current == (i + 1) % 10
        assert mem.count == (i + 1 if i < 10 else 10)

    print(np.mean(mem.observations, axis=(1, 2, 3)))
    print(mem.actions)
    print(mem.dones)

    obs, actions, rewards, dones, next_obs = mem.sample(4)
    for i in range(len(obs)):
        assert obs.shape == (4, 84, 84, 4)
        assert actions.shape == (4,)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert next_obs.shape == (4, 84, 84, 4)
        assert np.mean(obs[i, :, :, 3]) == actions[i]
        assert np.mean(obs[i, :, :, 2]) == actions[i] - 1
        assert np.mean(obs[i, :, :, 1]) == actions[i] - 2
        assert np.mean(obs[i, :, :, 0]) == actions[i] - 3
        assert np.mean(next_obs[i, :, :, 3]) == actions[i] + 1, str(np.mean(next_obs[i, :, :, 3])) + ' != ' + str(actions[i] + 1)
        assert np.mean(next_obs[i, :, :, 2]) == actions[i]
        assert np.mean(next_obs[i, :, :, 1]) == actions[i] - 1
        assert np.mean(next_obs[i, :, :, 0]) == actions[i] - 2
        assert rewards[i] == actions[i]
        assert dones[i] == (actions[i] % 17 == 0)
