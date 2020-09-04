from copy import deepcopy
import numpy as np
import time

import torch
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

"""
Actor and Critic agents used in various RL algorithms
"""

def mlp(layer_sizes, hidden_activation, final_activation, batchnorm=True):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            # if batchnorm:
            #     layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
            layers.append(hidden_activation())
        else:
            layers.append(final_activation())
    return nn.Sequential(*layers)


class NormalDistActor(nn.Module):
    """
    A stochastic policy for use in on-policy methods
    forward method: returns Normal dist and logprob of an action if passed
    model: N(mu, sigma) with MLP for mu, log(sigma) is a tunable parameter
    Input: observation/state
    """

    def __init__(self, layer_sizes_mu, act_dim, act_low, act_high, activation):
        super().__init__()
        self.mu_net = mlp(layer_sizes_mu, activation, nn.Identity)
        log_sigma = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_sigma = torch.nn.Parameter(torch.as_tensor(log_sigma))

    def distribution(self, obs):
        mu = self.mu_net(obs)
        sigma = torch.exp(self.log_sigma)
        return Normal(mu, sigma)

    def _logprob_from_distr(self, pi, act):
        # Need sum for a Normal distribution
        return pi.log_prob(act).sum(axis=-1)

    def forward(self, obs, act=None):
        pi = self._distribution(obs)
        logprob = None
        if act is not None:
            logprob = self._logprob_from_distr(pi, act)
        return pi, logprob


class ContinuousEstimator(nn.Module):
    """
    Generic MLP object to output continuous value(s).
    Can be used for:
        - V function (input: s, output: exp return)
        - Q function (input: (s, a), output: exp return)
        - deterministic policy (input: s, output: a)
    Layer sizes passed as argument.
    Input dimension: layer_sizes[0]
    Output dimension: layer_sizes[-1] (should be 1 for V,Q)
    """

    def __init__(self, layer_sizes, activation, final_activation=nn.Identity, **kwargs):
        super().__init__()
        self.net = mlp(layer_sizes, activation, final_activation)

    def forward(self, x):
        # x should contain [obs, act] for off-policy methods
        output = self.net(x)
        return torch.squeeze(output, -1)  # Critical to ensure v has right shape.


class BoundedDeterministicActor(nn.Module):
    """
    MLP net for actor in bounded continuous action space.
    Returns deterministic action.
    Layer sizes passed as argument.
    Input dimension: layer_sizes[0]
    Output dimension: layer_sizes[-1] (should be 1 for V,Q)
    """

    def __init__(self, layer_sizes, activation, low, high, **kwargs):
        super().__init__()
        self.low = torch.as_tensor(low)
        self.width = torch.as_tensor(high - low)
        self.net = mlp(layer_sizes, activation, nn.Tanh)

    def forward(self, x):
        output = (self.net(x) + 1) * self.width / 2 + self.low
        return output


class BoundedStochasticActor(nn.Module):
    """
    Produces a squashed Normal distribution for one var from a MLP for mu and sigma.
    """

    def __init__(
        self,
        layer_sizes,
        act_dim,
        act_low,
        act_high,
        activation=nn.ReLU,
        log_sigma_min=-20,
        log_sigma_max=2,
    ):
        super().__init__()
        self.act_low = torch.as_tensor(act_low)
        self.act_width = torch.as_tensor(act_high - act_low)
        self.shared_net = mlp(layer_sizes, activation, activation)
        self.mu_layer = nn.Linear(layer_sizes[-1], act_dim, activation)
        self.log_sigma_layer = nn.Linear(layer_sizes[-1], act_dim, activation)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, obs, deterministic=False, get_logprob=False):
        shared = self.shared_net(obs)
        mu = self.mu_layer(shared)
        log_sigma = self.log_sigma_layer(shared)
        log_sigma = torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma = torch.exp(log_sigma)
        pi = Normal(mu, sigma)
        if deterministic:
            # For evaluating performance at end of epoch, not for data collection
            act = mu
        else:
            act = pi.rsample()
        logprob = None
        if get_logprob:
            logprob = pi.log_prob(act).sum(axis=-1)
            # Convert pdf due to tanh transform
            logprob -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(axis=1)
        act = torch.tanh(act)
        act = (act + 1) * self.act_width / 2 + self.act_low
        return act, logprob


class GaussianActorCritic(nn.Module):
    """
    Contains an actor (to produce policy and act)
    and a critic (to estimate value function)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_mu=(64, 64),
        hidden_layers_v=(64, 64),
        activation=nn.Tanh,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layer_sizes_mu = [obs_dim] + list(hidden_layers_mu) + [act_dim]
        layer_sizes_v = [obs_dim] + list(hidden_layers_v) + [1]
        self.pi = NormalDistActor(
            layer_sizes_mu=layer_sizes_mu, act_dim=act_dim, activation=activation,
        )
        self.v = ContinuousEstimator(layer_sizes_v=layer_sizes_v, activation=activation)
        self.act_low = action_space.low
        self.act_high = action_space.high

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi.distribution(obs)
            act = pi.sample()
            logprob = pi.log_prob(act)
            logprob = logprob.sum(axis=-1)
            act = act.clamp(self.act_low, self.act_high)
            val = self.v(obs)
        return act.numpy(), val.numpy(), logprob.numpy()

    def act(self, obs):
        return self.step(obs)[0]


class DDPGAgent(nn.Module):
    """
    Agent to be used in DDPG.
    Contains:
    - estimated Q*(s,a,)
    - policy

    """

    def __init__(
        self,
        obs_dim=None,
        obs_space=None,
        action_space=None,
        hidden_layers_mu=(64, 64),
        hidden_layers_q=(64, 64),
        activation=nn.ReLU,
        final_activation=nn.Tanh,
        noise_std=0.1,
        pi_lr=1e-3,
        q_lr=1e-3,
        polyak=0.995,
        gamma=0.99,
        **kwargs
    ):
        super().__init__()
        if obs_dim is None:
            obs_dim = obs_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_low = action_space.low
        self.act_high = action_space.high

        layer_sizes_mu = [obs_dim] + list(hidden_layers_mu) + [self.act_dim]
        layer_sizes_q = [obs_dim + self.act_dim] + list(hidden_layers_q) + [1]

        self.noise_std = noise_std
        self.polyak = polyak
        self.gamma = gamma

        self.pi = BoundedDeterministicActor(
            layer_sizes=layer_sizes_mu,
            activation=activation,
            final_activation=final_activation,
            low=self.act_low,
            high=self.act_high,
            **kwargs
        )
        self.q = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        self.q_optimizer = Adam(self.q.parameters(), lr=q_lr)

        self.target = deepcopy(self)
        for p in self.target.parameters():
            p.requires_grad = False

    def act(self, obs, noise=False):
        """Return noisy action as numpy array, **without computing grads**"""
        # TO DO: fix how noise and clipping are handled for multiple dimensions.
        with torch.no_grad():
            act = self.pi(obs).numpy()
            if noise:
                act += self.noise_std * np.random.randn(self.act_dim)
            act = np.clip(act, self.act_low[0], self.act_high[0])
        return act

    def update_pi(self, data=None):
        # Freeze Q params during policy update to save time
        for p in self.q.parameters():
            p.requires_grad = False
        self.pi_optimizer.zero_grad()
        o = data["obs"]
        a = self.pi(o)
        pi_loss = -self.q(torch.cat((o, a), dim=-1)).mean()
        pi_loss.backward()
        self.pi_optimizer.step()
        # Unfreeze Q params after policy update
        for p in self.q.parameters():
            p.requires_grad = True
        return pi_loss

    def update_q(self, data=None, agent=None):
        r, o_next, d = data["reward"], data["obs_next"], data["done"]
        if agent is not None:
            r = r[:, agent]
        self.q_optimizer.zero_grad()
        with torch.no_grad():
            a_next = self.target.pi(o_next)
            q_target = self.target.q(torch.cat((o_next, a_next), dim=-1))
            q_target = r + self.gamma * (1 - d) * q_target
        o, a = data["obs"], data["act"]
        if agent is not None:
            a = a[:, agent]
        q = self.q(torch.cat((o, a), dim=-1))
        q_loss_info = {"QVals": q.detach().numpy()}
        q_loss = ((q - q_target) ** 2).mean()
        q_loss.backward()
        self.q_optimizer.step()
        return q_loss, q_loss_info

    def update_target(self):
        with torch.no_grad():
            # Use in place method from Spinning Up, faster than creating a new state_dict
            for p, p_target in zip(self.pi.parameters(), self.target.pi.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)
            for p, p_target in zip(self.q.parameters(), self.target.q.parameters()):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)

    def update(self, data, logger=None, **kwargs):
        pi_loss = self.update_pi(data=data)
        q_loss, q_loss_info = self.update_q(data=data, **kwargs)
        self.update_target()
        # Record things
        if logger is not None:
            logger.store(LossQ=q_loss.item(), LossPi=pi_loss.item(), **q_loss_info)
        return pi_loss, q_loss, q_loss_info


class TD3Agent(nn.Module):
    """
    Agent to be used in TD3.
    Contains:
    - estimated Q*(s,a,)
    - policy

    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_mu=(256, 256),
        hidden_layers_q=(256, 256),
        activation=nn.ReLU,
        final_activation=nn.Tanh,
        noise_std=0.1,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_low = action_space.low
        self.act_high = action_space.high

        layer_sizes_mu = [obs_dim] + list(hidden_layers_mu) + [act_dim]
        layer_sizes_q = [obs_dim + act_dim] + list(hidden_layers_q) + [1]
        self.noise_std = noise_std
        self.pi = BoundedDeterministicActor(
            layer_sizes=layer_sizes_mu,
            activation=activation,
            final_activation=final_activation,
            low=self.act_low,
            high=self.act_high,
            **kwargs
        )
        self.q1 = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )
        self.q2 = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )

    def act(self, obs, noise=False):
        """Return noisy action as numpy array, **without computing grads**"""
        # TO DO: fix how noise and clipping are handled for multiple dimensions.
        with torch.no_grad():
            act = self.pi(obs).numpy()
            if noise:
                act += self.noise_std * np.random.randn(self.act_dim)
            act = np.clip(act, self.act_low[0], self.act_high[0])
        return act


class SACAgent(nn.Module):
    """
    Agent to be used in SAC.
    Contains:
    - stochastic policy (bounded by tanh)
    - estimated Q*(s,a,)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_pi=(256, 256),
        hidden_layers_q=(256, 256),
        activation=nn.ReLU,
        **kwargs
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layer_sizes_pi = [obs_dim] + list(hidden_layers_pi)
        layer_sizes_q = [obs_dim + act_dim] + list(hidden_layers_q) + [1]
        self.pi = BoundedStochasticActor(
            layer_sizes_pi,
            act_dim,
            action_space.low,
            action_space.high,
            activation=activation,
            **kwargs
        )
        self.q1 = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )
        self.q2 = ContinuousEstimator(
            layer_sizes=layer_sizes_q, activation=activation, **kwargs
        )

    def act(self, obs, deterministic=False):
        """Return noisy action as numpy array, **without computing grads**"""
        with torch.no_grad():
            act, _ = self.pi(obs, deterministic=deterministic)
        return act.numpy()


