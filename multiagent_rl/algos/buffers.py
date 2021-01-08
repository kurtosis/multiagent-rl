import numpy as np

import torch

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
    if shape2 is None:
        return (shape1,)
    elif np.isscalar(shape2):
        return (shape1, shape2)
    else:
        return (shape1, *shape2)


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
        of episode (b/c they depend on future rewards):
        - Advantage (for GAE)
        - Return (using reward-to-go)
        Compute both of those here and save to buffer.
        Update start index for next episode.
        """
        # note location of current episode in buffer
        path_slice = slice(self.path_start, self.ptr)
        # get rewards and values of current episode, append the last step value
        rewards = np.append(self.reward[path_slice], last_v)
        values = np.append(self.v[path_slice], last_v)
        # compute advantage fn A(s_t,a_t) for each step in episode using GAE
        # write this to the buffer in the location of this episode
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        self.adv[path_slice] = discount_cumsum(deltas, self.gamma * self.lamb)
        # compute rewards to go
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


class RDPGBuffer:
    """
    Stores completed episodes for use in RDPG with recurrent networks. Buffer is a list
    of (complete) episodes. Once buffer is full, older episodes are overwritten.
    """

    def __init__(self, max_size):
        self.ptr = 0
        # self.path_start = 0
        self.max_size = max_size
        self.current_start = self.ptr
        self.current_episode = []
        self.episodes = [None] * max_size
        self.filled_size = 0
        self.full = False

    def store(self, obs, act, reward, obs_next, done):
        """Add latest step variables to current trajectory. If done, add trajectory
        to buffer and reset. Loop pointer back to 0 when buffer is full."""
        self.current_episode.append(
            dict(obs=obs, act=act, reward=reward, obs_next=obs_next, done=done)
        )
        if done == 1:
            self.episodes[self.ptr] = self.current_episode
            self.current_episode = []
            self.ptr += 1
            if not self.full:
                self.filled_size += 1
            if self.ptr == self.max_size:
                self.ptr = 0
                self.full = True

    def reshape_samples(self, samples):
        vars = ["obs", "act", "reward", "obs_next", "done"]
        data = {v : torch.as_tensor([[x[v] for x in ep] for ep in samples], dtype=torch.float32) for v in vars}
        for v in vars:
            data[v] = data[v].transpose(0, 1)
        return data

    def sample_episodes(self, sample_size=100):
        sample_indexes = np.random.randint(0, self.filled_size, sample_size)
        samples = [self.episodes[i] for i in sample_indexes]
        return self.reshape_samples(samples)

    def get_latest_episodes(self, sample_size=100):
        end = self.ptr
        if self.ptr >= sample_size:
            start = self.ptr - sample_size
            samples = self.episodes[start:end]
            self.reshape_samples(samples)

        return samples

    def clear_current_episode(self):
        self.current_episode = []


# For off-policy methods
class TransitionBuffer:
    def __init__(self, obs_dim, act_dim, max_size):
        self.obs = np.zeros(merge_shape(max_size, obs_dim), dtype=np.float32)
        self.act = np.zeros(merge_shape(max_size, act_dim), dtype=np.float32)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.obs_next = np.zeros(merge_shape(max_size, obs_dim), dtype=np.float32)
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

    def get(self, sample_size=100):
        """
        Return needed variables (as tensors) over episodes in buffer.
        Reset pointers for next epoch.
        """
        # return needed variables as a dictionary
        sample_indexes = np.random.randint(0, self.filled_size, sample_size)
        data = {
            "obs": torch.as_tensor(self.obs[sample_indexes], dtype=torch.float32),
            "act": torch.as_tensor(self.act[sample_indexes], dtype=torch.float32),
            "reward": torch.as_tensor(self.reward[sample_indexes], dtype=torch.float32),
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

    def get(self, sample_size=100):
        """
        Return needed variables (as tensors) over episodes in buffer.
        Reset pointers for next epoch.
        """
        # return needed variables as a dictionary
        sample_indexes = np.random.randint(0, self.filled_size, sample_size)
        data = {
            "obs": torch.as_tensor(self.obs[sample_indexes], dtype=torch.float32),
            "act": torch.as_tensor(self.act[sample_indexes], dtype=torch.float32),
            "reward": torch.as_tensor(self.reward[sample_indexes], dtype=torch.float32),
            "obs_next": torch.as_tensor(
                self.obs_next[sample_indexes], dtype=torch.float32
            ),
            "done": torch.as_tensor(self.done[sample_indexes], dtype=torch.float32),
        }
        return data
