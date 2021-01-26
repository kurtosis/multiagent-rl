import numpy as np

import torch
from torch.nn.functional import pad

"""
Classes for buffers that are used by policy-gradient and deep-Q training.
"""


def discount_cumsum(x, discount):
    """Compute cumsum with discounting used in GAE (generalized adv estimn).
    (My implementation)"""
    discounts = [discount ** ll for ll in range(len(x))]
    disc_seqs = [discounts] + [discounts[:-i] for i in range(1, len(x))]
    return np.array([np.dot(x[i:], disc_seqs[i]) for i in range(len(x))])


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


# For on-policy VPG/PPO
class TrajectoryBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lamb=0.95):
        self.obs = np.zeros(merge_shape(size, obs_dim), dtype=np.float32)
        self.act = np.zeros(merge_shape(size, act_dim), dtype=np.float32)
        self.adv = np.zeros(size, dtype=np.float32)
        self.reward = np.zeros(size, dtype=np.float32)
        self.ret = np.zeros(size, dtype=np.float32)
        self.v = np.zeros(size, dtype=np.float32)
        self.logprob = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.lamb = lamb
        self.ptr = 0
        self.path_start = 0
        self.max_size = size

    def store(self, obs, act, reward, val, logprob):
        """Add current step variables to buffer."""
        assert self.ptr < self.max_size
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.reward[self.ptr] = reward
        self.v[self.ptr] = val
        self.logprob[self.ptr] = logprob
        self.ptr += 1

    def finish_path(self, last_v=0):
        """
        We've logged most variables at each step in episode.
        There are two vars that can only be computed at end
        of episode (b/c they depend on future reward):
        - Advantage (for GAE)
        - Return (using reward-to-go)
        Compute both of those here and save to buffer.
        Update start index for next episode.
        """
        # note location of current episode in buffer
        path_slice = slice(self.path_start, self.ptr)
        # get reward and values of current episode, append the last step value
        rewards = np.append(self.reward[path_slice], last_v)
        values = np.append(self.v[path_slice], last_v)
        # compute advantage fn A(s_t,a_t) for each step in episode using GAE
        # write this to the buffer in the location of this episode
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv[path_slice] = discount_cumsum(deltas, self.gamma * self.lamb)
        # compute reward to go
        self.ret[path_slice] = discount_cumsum(rewards, self.gamma)[:-1]
        # Update start index for next episode
        self.path_start = self.ptr

    def get(self):
        """
        Return needed variables (as tensors) over episodes in buffer.
        Note that advantage is normalized first.
        Reset pointers for next epoch.
        """
        # can only get data when buffer is full
        assert self.ptr == self.max_size
        # reset pointers for next epoch
        self.ptr = 0
        self.path_start = 0
        # Normalize adv for GAE
        adv_mean = self.adv.mean()
        adv_std = self.adv.std()
        self.adv = (self.adv - adv_mean) / adv_std
        # return needed variables as a dictionary
        data = {
            "obs": self.obs,
            "act": self.act,
            "adv": self.adv,
            "ret": self.ret,
            "logprob": self.logprob,
        }
        data = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}
        return data


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(merge_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(merge_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(merge_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done_buf[idxs],
        )
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class EpisodeBuffer:
    """
    Stores completed episodes for use in RDPG with recurrent networks. Each variable
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


# For off-policy methods
class TransitionBuffer:
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


# For off-policy methods
class MultiagentTransitionBuffer:
    def __init__(self, obs_dim, act_dim, num_agents, max_size):
        self.obs = np.zeros((max_size, num_agents, obs_dim), dtype=np.float32)
        self.act = np.zeros((max_size, num_agents, act_dim), dtype=np.float32)
        self.reward = np.zeros((max_size, num_agents), dtype=np.float32)
        self.obs_next = np.zeros((max_size, num_agents, obs_dim), dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.float32)
        self.ptr = 0
        self.max_size = max_size
        self.filled_size = 0
        self.full = False

    def store(self, obs, act, reward, obs_next, done):
        """Add current step variables to buffer."""
        # Cycle through buffer, overwriting oldest entry.
        # Note that buffer is never flushed, unlike on-policy methods.
        if self.ptr >= self.max_size:
            self.ptr = self.ptr % self.max_size
            self.full = True
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.reward[self.ptr] = reward
        self.obs_next[self.ptr] = obs_next
        self.done[self.ptr] = done
        self.ptr += 1
        if not self.full:
            self.filled_size += 1

    def get(self, batch_size=100):
        """
        Return needed variables (as tensors) over episodes in buffer.
        Reset pointers for next epoch.
        """
        # return needed variables as a dictionary
        sample_indexes = np.random.randint(0, self.filled_size, batch_size)
        data = {
            "obs": torch.as_tensor(self.obs[sample_indexes], dtype=torch.float32),
            "act": torch.as_tensor(self.act[sample_indexes], dtype=torch.float32),
            "rwd": torch.as_tensor(self.reward[sample_indexes], dtype=torch.float32),
            "obs_next": torch.as_tensor(
                self.obs_next[sample_indexes], dtype=torch.float32
            ),
            "done": torch.as_tensor(self.done[sample_indexes], dtype=torch.float32),
        }
        return data
