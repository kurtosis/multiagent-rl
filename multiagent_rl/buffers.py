import numpy as np

import torch
from torch.nn.functional import pad


def merge_shape(shape1, shape2=None):
    if np.isscalar(shape1):
        if shape2 is None:
            return (shape1,)
        elif np.isscalar(shape2):
            return shape1, shape2
        else:
            return (shape1, *shape2)
    else:
        if shape2 is None:
            return shape1
        elif np.isscalar(shape2):
            return (*shape1, shape2)
        else:
            return (*shape1, *shape2)


class EpisodeBuffer:
    """
    Stores completed episodes for training recurrent actor/critic networks. Each variable
    is stored as a tensor for fast sampling. Once buffer is full, older episodes are overwritten.
    """

    def __init__(self, obs_dim, act_dim, max_episode_len, max_episodes):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_episode_len = max_episode_len
        self.max_episodes = max_episodes
        self.ptr_turn = 0
        self.ptr_ep = 0
        self.data = {
            "obs": np.zeros(
                merge_shape((max_episode_len, max_episodes), obs_dim), dtype=np.float32
            ),
            "act": np.zeros(
                merge_shape((max_episode_len, max_episodes), act_dim), dtype=np.float32
            ),
            "rwd": np.zeros((max_episode_len, max_episodes, 1), dtype=np.float32),
            "done": np.ones((max_episode_len, max_episodes, 1), dtype=np.float32),
        }
        self.current_episode = {
            "obs": np.zeros(
                merge_shape((max_episode_len, 1), obs_dim), dtype=np.float32
            ),
            "act": np.zeros(
                merge_shape((max_episode_len, 1), act_dim), dtype=np.float32
            ),
            "rwd": np.zeros((max_episode_len, 1, 1), dtype=np.float32),
            "done": np.ones((max_episode_len, 1, 1), dtype=np.float32),
        }
        self.filled_size = 0
        self.full = False

    def store(self, obs, act, rwd, obs_next, done):
        """Add latest step variables to current trajectory. If done, add trajectory
        to buffer and reset. Loop pointer back to 0 when buffer is full."""
        self.current_episode["obs"][self.ptr_turn, 0, :] = obs
        self.current_episode["act"][self.ptr_turn, 0, :] = act
        self.current_episode["rwd"][self.ptr_turn, 0, :] = rwd
        self.current_episode["done"][self.ptr_turn, 0, :] = done
        self.ptr_turn += 1

        if done == 1:
            for v in self.data:
                self.data[v][:, self.ptr_ep, :] = self.current_episode[v][:, 0, :]
            self.ptr_ep += 1
            self.ptr_turn = 0
            if not self.full:
                self.filled_size += 1
            if self.ptr_ep == self.max_episodes:
                self.ptr_ep = 0
                self.full = True

    def sample_episodes(self, batch_size=100):
        sample_indexes = np.random.randint(0, self.filled_size, batch_size)
        samples = {
            v: torch.as_tensor(self.data[v][:, sample_indexes, :]) for v in self.data
        }
        samples["obs_next"] = samples["obs"][1:, :, :]
        samples["obs_next"] = pad(
            samples["obs_next"], (0, 0, 0, 0, 0, 1), "constant", 0
        )
        return samples

    def clear_current_episode(self):
        self.current_episode = {
            "obs": np.zeros_like(self.current_episode["obs"], dtype=np.float32),
            "act": np.zeros_like(self.current_episode["act"], dtype=np.float32),
            "rwd": np.zeros_like(self.current_episode["rwd"], dtype=np.float32),
            "done": np.ones_like(self.current_episode["done"], dtype=np.float32),
        }


class TransitionBuffer:
    """
    Stores individual transitions for training non-recurrent actor/critic networks.
    Once buffer is full, older episodes are overwritten.
    """

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs = np.zeros(merge_shape(max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros(merge_shape(max_size, act_dim), dtype=np.float32)
        self.rwd = np.zeros(max_size, dtype=np.float32)
        self.obs_next = np.zeros(merge_shape(max_size, obs_dim), dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.max_size = max_size
        self.filled_size = 0
        self.full = False

    def store(self, obs, act, rwd, obs_next, done):
        """Add current step variables to buffer."""
        # Cycle through buffer, overwriting oldest entry.
        # Note that buffer is never flushed, unlike on-policy methods.
        if self.ptr >= self.max_size:
            self.ptr = self.ptr % self.max_size
            self.full = True
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rwd[self.ptr] = rwd
        self.obs_next[self.ptr] = obs_next
        self.done[self.ptr] = done
        self.ptr += 1
        if not self.full:
            self.filled_size += 1

    def sample_batch(self, batch_size=128):
        """
        Return needed variables (as tensors) over episodes in buffer.
        Reset pointers for next epoch.
        """
        # return needed variables as a dictionary
        sample_indexes = np.random.randint(0, self.filled_size, batch_size)
        data = {
            "obs": torch.as_tensor(self.obs[sample_indexes], dtype=torch.float32),
            "act": torch.as_tensor(self.act[sample_indexes], dtype=torch.float32),
            "rwd": torch.as_tensor(self.rwd[sample_indexes], dtype=torch.float32),
            "obs_next": torch.as_tensor(
                self.obs_next[sample_indexes], dtype=torch.float32
            ),
            "done": torch.as_tensor(self.done[sample_indexes], dtype=torch.float32),
        }
        return data
