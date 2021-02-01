from copy import deepcopy
from itertools import chain
import numpy as np

import torch
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

from multiagent_rl.algos.buffers import EpisodeBuffer


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


#########################################
# Actors
#########################################
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

    def _distribution(self, obs):
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
        self.mu_layer = nn.Linear(layer_sizes[-1], act_dim)
        self.log_sigma_layer = nn.Linear(layer_sizes[-1], act_dim)
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, obs, deterministic=False, get_logprob=True):
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
            # Changed sum axis to work with EpisodeBuffer. Need to ensure this is correct.
            # logprob -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(axis=1)
            logprob -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(axis=-1)
        else:
            logprob = None
        act = torch.tanh(act)
        act = (act + 1) * self.act_width / 2 + self.act_low
        return act, logprob


class LSTMDeterministicActor(nn.Module):
    """
    MLP net for actor in bounded continuous action space.
    Returns deterministic action.
    Layer sizes passed as argument.
    Input dimension: layer_sizes[0]
    Output dimension: layer_sizes[-1] (should be 1 for V,Q)
    """

    # def __init__(self, layer_sizes, activation, low, high, **kwargs):
    def __init__(self, input_size, hidden_size, action_size, low, high, **kwargs):
        super().__init__()
        self.low = torch.as_tensor(low)
        self.width = torch.as_tensor(high - low)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.action_mlp = mlp([hidden_size, action_size], nn.Identity, nn.Tanh)
        self.h = torch.zeros((1, 1, self.hidden_size))
        self.c = torch.zeros((1, 1, self.hidden_size))

    def forward(self, input, action_only=True):
        """The Agent must distinguish between generating actions sequentially (and maintaining hidden
         state) and training on a batch of trajectories."""
        if len(input.shape) == 3:
            batch_size = input.shape[1]
            h = torch.zeros((1, batch_size, self.hidden_size))
            c = torch.zeros((1, batch_size, self.hidden_size))
            lstm_out, (h, c) = self.lstm(input, (h, c))
        else:
            # TO DO: what should the input dims be exactly? Are there edge cases?
            lstm_out, (h, c) = self.lstm(
                input.view(-1, 1, len(input)), (self.h, self.c)
            )
            self.h, self.c = h, c
        x = self.action_mlp(lstm_out)
        action_out = (x + 1) * self.width / 2 + self.low
        if action_only:
            return action_out
        else:
            return action_out, (h, c)

    def reset_state(self):
        self.h = torch.zeros_like(self.h)
        self.c = torch.zeros_like(self.c)


class LSTMStochasticActor(nn.Module):
    """
    LSTM-based stochastic Actor for RSAC (Recurrent Soft Actor-Critic).
    Actions are generated by sampling from a normal distribution then passing through tanh to yield
    values in the action space. If 'deterministic==True', mean values are used rather than sampling.

    Args:
        input_size : number of features in input (typically obs)
        hidden_size : number of hidden features in the LSTM
        layer_sizes : tuple of hidden layer sizes in the output MLP
        act_dim : number of features in output (i.e. the action)
        act_low/high : lists of lower and upper bounds of all output features
        activation : activation function used in the output MLP
        log_sigma_min/max : log_sigma is constrained to values in this range
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        layer_sizes,
        act_dim,
        act_low,
        act_high,
        activation=nn.ReLU,
        log_sigma_min=-20,
        log_sigma_max=2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.act_low = torch.as_tensor(act_low)
        self.act_width = torch.as_tensor(act_high - act_low)
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.shared_mlp = mlp(layer_sizes, activation, activation)
        self.mu_layer = nn.Linear(layer_sizes[-1], act_dim)
        self.log_sigma_layer = nn.Linear(layer_sizes[-1], act_dim)
        self.h = torch.zeros((1, 1, self.hidden_size))
        self.c = torch.zeros((1, 1, self.hidden_size))
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, input, deterministic=False, get_logprob=True):
        """The Agent must distinguish between generating actions sequentially (and maintaining hidden
         state) and training on a batch of trajectories."""
        if len(input.shape) == 3:
            batch_size = input.shape[1]
            h = torch.zeros((1, batch_size, self.hidden_size))
            c = torch.zeros((1, batch_size, self.hidden_size))
            lstm_out, _ = self.lstm(input, (h, c))
        else:
            lstm_out, (h, c) = self.lstm(
                input.view(-1, 1, len(input)), (self.h, self.c)
            )
            self.h, self.c = h, c
        shared = self.shared_mlp(lstm_out)
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
        if get_logprob:
            logprob = pi.log_prob(act).sum(axis=-1)
            # Convert pdf due to tanh transform
            # TODO: make sure axis=-1 is right
            logprob -= (2 * (np.log(2) - act - F.softplus(-2 * act))).sum(axis=-1)
        else:
            logprob = None
        act = torch.tanh(act)
        act = (act + 1) * self.act_width / 2 + self.act_low
        return act, logprob

    def reset_state(self):
        self.h = torch.zeros_like(self.h)
        self.c = torch.zeros_like(self.c)


#########################################
# Critics
#########################################
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

    def reset_state(self):
        pass


class LSTMEstimator(nn.Module):
    """
    LSTM-based Critic for RSAC (Recurrent Soft Actor-Critic).
    Can be used to represent Q((obs, act)) or V(obs). Output is deterministic (unlike RSAC Actor).

    Args:
        input_size : number of features in input (typically (obs, act) for Q).
        hidden_size : number of hidden features in the LSTM
        layer_sizes : tuple of hidden layer sizes in the output MLP
        activation : activation function used in the output MLP
        final_activation : final activation function in the output MLP
    """

    def __init__(
        self,
        input_size,
        hidden_size,
        layer_sizes,
        activation=nn.ReLU,
        final_activation=nn.Identity,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.value_mlp = mlp(layer_sizes, activation, final_activation)
        self.h = torch.zeros((1, 1, self.hidden_size))
        self.c = torch.zeros((1, 1, self.hidden_size))

    def forward(self, input, value_only=True):
        """The Agent must distinguish between generating actions sequentially (and maintaining hidden
         state) and training on a batch of trajectories."""
        if len(input.shape) == 3:
            batch_size = input.shape[1]
            h = torch.zeros((1, batch_size, self.hidden_size))
            c = torch.zeros((1, batch_size, self.hidden_size))
            lstm_out, (h, c) = self.lstm(input, (h, c))
        else:
            lstm_out, (h, c) = self.lstm(
                input.view(len(input), 1, -1), (self.h, self.c)
            )
            self.h, self.c = h, c
        value = self.value_mlp(lstm_out)
        value = value.squeeze()
        if value_only:
            return value
        else:
            return value, (h, c)

    def reset_state(self):
        self.h = torch.zeros_like(self.h)
        self.c = torch.zeros_like(self.c)


#########################################
# Actor-Critic Joint models
#########################################
class LSTMJoinedActorCritic(nn.Module):
    """
    MLP net for actor + value in bounded continuous action space.
    Returns deterministic action.
    Layer sizes passed as argument.
    Input dimension: layer_sizes[0]
    Output dimension: layer_sizes[-1] (should be 1 for V,Q)
    """

    # def __init__(self, layer_sizes, activation, low, high, **kwargs):
    def __init__(
        self,
        input_size,
        hidden_size,
        action_size,
        low,
        high,
        activation=nn.ReLU,
        final_activation=nn.Identity,
        **kwargs,
    ):
        super().__init__()
        # self.net = mlp(layer_sizes, activation, nn.Tanh)

        self.low = torch.as_tensor(low)
        self.width = torch.as_tensor(high - low)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.action_mlp = mlp([hidden_size, action_size], nn.Identity, nn.Tanh)
        self.value_mlp = mlp([hidden_size, 1], activation, final_activation)
        self.h = torch.zeros((1, 1, self.hidden_size))
        self.c = torch.zeros((1, 1, self.hidden_size))

    def forward(self, input, h=None, c=None):
        # if h is None:
        #     h = torch.zeros(self.hidden_size)
        # if c is None:
        #     c = torch.zeros(self.hidden_size)
        lstm_out, (h, c) = self.lstm(input.view(len(input), 1, -1), (self.h, self.c))
        self.h, self.c = h, c
        x = self.action_mlp(lstm_out)
        action_out = (x + 1) * self.width / 2 + self.low
        value = self.value_mlp(lstm_out)
        return action_out, value, (h, c)


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
        **kwargs,
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


#########################################
# Agents
#########################################
class DDPGAgent(nn.Module):
    """
    Agent to be used in DDPG.
    Contains:
    - estimated Q*(s,a,)
    - policy

    """

    def __init__(
        self,
        obs_space=None,
        action_space=None,
        obs_dim=None,
        hidden_layers_mu=(256, 256),
        hidden_layers_q=(256, 256),
        activation=nn.ReLU,
        final_activation=nn.Tanh,
        noise_std=0.1,
        pi_lr=1e-3,
        q_lr=1e-3,
        polyak=0.995,
        gamma=0.99,
        **kwargs,
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
            **kwargs,
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
        r, o_next, d = data["rwd"], data["obs_next"], data["done"]
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
            logger.store(**q_loss_info)
        return pi_loss, q_loss, q_loss_info

    def reset_state(self):
        pass


class TD3Agent(nn.Module):
    """
    Agent to be used in TD3.
    Contains:
    - estimated Q*(s,a,)
    - policy

    """

    def __init__(
        self,
        obs_space,
        action_space,
        hidden_layers_mu=(256, 256),
        hidden_layers_q=(256, 256),
        activation=nn.ReLU,
        final_activation=nn.Tanh,
        noise_std=0.1,
        **kwargs,
    ):
        super().__init__()
        obs_dim = obs_space.shape[0]
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
            **kwargs,
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

    def reset_state(self):
        pass


class SACAgent(nn.Module):
    """
    Agent to be used in SAC.
    Contains:
    - stochastic policy (bounded by tanh)
    - estimated Q*(s,a,)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        hidden_layers_pi=(256, 256),
        hidden_layers_q=(256, 256),
        activation=nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        obs_dim = obs_space.shape[0]
        act_dim = action_space.shape[0]
        layer_sizes_pi = [obs_dim] + list(hidden_layers_pi)
        layer_sizes_q = [obs_dim + act_dim] + list(hidden_layers_q) + [1]
        self.pi = BoundedStochasticActor(
            layer_sizes_pi,
            act_dim,
            action_space.low,
            action_space.high,
            activation=activation,
            **kwargs,
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
            act, _ = self.pi(obs, deterministic=deterministic, get_logprob=False)
        return act.numpy()

    def reset_state(self):
        pass


class RDPGAgent(nn.Module):
    """
    Agent to be used in RDPG.
    Contains:
    - estimated Q*(s,a,)
    - policy
    """

    def __init__(
        self,
        obs_space=None,
        action_space=None,
        obs_dim=None,
        hidden_size=64,
        noise_std=0.1,
        pi_lr=1e-3,
        q_lr=1e-3,
        polyak=0.995,
        gamma=0.99,
        q_fn="LSTMEstimator",
        **kwargs,
    ):
        super().__init__()
        if obs_dim is None:
            obs_dim = obs_space.shape[0]
        self.act_dim = action_space.shape[0]
        self.act_low = action_space.low
        self.act_high = action_space.high

        self.noise_std = noise_std
        self.polyak = polyak
        self.gamma = gamma

        self.pi = LSTMDeterministicActor(
            input_size=obs_dim,
            hidden_size=hidden_size,
            action_size=self.act_dim,
            low=self.act_low,
            high=self.act_high,
            **kwargs,
        )

        # This is a quick hack to test RDPG with a non-LSTM Q function
        if q_fn == "ContinuousEstimator":
            hidden_layers_q = (64, 64)
            activation = nn.ReLU
            layer_sizes_q = [obs_dim + self.act_dim] + list(hidden_layers_q) + [1]
            self.q = ContinuousEstimator(
                layer_sizes=layer_sizes_q, activation=activation, **kwargs
            )
        else:
            self.q = LSTMEstimator(
                input_size=obs_dim + self.act_dim,
                hidden_size=hidden_size,
                action_size=self.act_dim,
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
            act = np.squeeze(act, (0, 1))
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
        a_proposed = self.pi(o)
        pi_loss = -self.q(torch.cat((o, a_proposed), dim=-1)).mean()
        pi_loss.backward()
        self.pi_optimizer.step()
        # Unfreeze Q params after policy update
        for p in self.q.parameters():
            p.requires_grad = True
        return pi_loss

    def update_q(self, data=None, agent=None):
        r, o_next, d = data["rwd"], data["obs_next"], data["done"]
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
            logger.store(**q_loss_info)
        return pi_loss, q_loss, q_loss_info

    def reset_state(self):
        self.pi.reset_state()
        self.q.reset_state()


class RSACAgent(nn.Module):
    """
    Recurrent (LSTM) Agent for SAC (Soft Actor-Critic).

    This class is designed for multi-agent RL. In addition to the Actor-Critic module it contains a number of
    attributes and methods that might typically be instantiated outside the Agent class in single-agent RL.
    For clarity, some key components are:

    Key Attributes:
        pi : the (LSTM-based) actor
        q1, q2 : the (LSTM-based) critics
        q1_targ, q2_targ : lagged copies of the critics used in updating
        buffer : a buffer of past episodes used for updating the Agent
        pi/q/alpha_optimizer : optimizers for all updates performed by the Agent

    Key Methods:
        act : return action given obs, based on pi
        store_to_buffer : add an interaction to buffer
        update : update pi, q1, q2, and alpha
        reset_state : reset pi, q1, and q2 modules for the start of a new episode

    Args:
        obs_space : the observation space, typically defined by the environment
        action_space : the action space, typically defined by the environment
        hidden_size_pi : number of hidden features in the pi LSTM
        hidden_size_q : number of hidden features in the Q LSTM
        mlp_layers_pi : tuple of hidden layer sizes in pi output MLP
        mlp_layers_q : tuple of hidden layer sizes in Q output MLP
        activation : activation function used in Actor and Critic
        max_ep_len : max episode length, used in buffer size
        max_buf_len : max number of episodes buffer can hold
        pi_lr : learning rate for pi updates
        q_lr : learning rate for Q updates
        a_lr : learning rate for alpha updates
        gamma : discount factor (between 0 and 1) for Q updates
        polyak : interpolation factor for Q target updates (between 0 and 1, typically close to 1)
        alpha : entropy regularization coefficient (larger values penalize low-entropy pi)
        update_alpha_after : number of env interactions to run before updating alpha
        target_entropy : controls the min value that alpha is reduced to during training.
            (typically negative, lower values cause alpha to be reduced more)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        hidden_size_pi=256,
        hidden_size_q=256,
        mlp_layers_pi=(256, 256),
        mlp_layers_q=(256, 256),
        activation=nn.ReLU,
        max_ep_len=10,
        max_buf_len=10000,
        pi_lr=1e-3,
        q_lr=1e-3,
        a_lr=1e-3,
        gamma=0.99,
        polyak=0.995,
        alpha=0.05,
        update_alpha_after=5000,
        target_entropy=-4.0,
        **kwargs,
    ):
        super().__init__()
        self.obs_dim = obs_space.shape[0]
        self.act_dim = action_space.shape[0]
        layer_sizes_pi = [hidden_size_pi] + list(mlp_layers_pi)
        layer_sizes_q = [hidden_size_q] + list(mlp_layers_q) + [1]

        self.pi = LSTMStochasticActor(
            self.obs_dim,
            hidden_size_pi,
            layer_sizes_pi,
            self.act_dim,
            action_space.low,
            action_space.high,
            activation=activation,
        )
        self.q1 = LSTMEstimator(
            input_size=self.obs_dim + self.act_dim,
            hidden_size=hidden_size_q,
            layer_sizes=layer_sizes_q,
            activation=activation,
        )
        self.q2 = LSTMEstimator(
            input_size=self.obs_dim + self.act_dim,
            hidden_size=hidden_size_q,
            layer_sizes=layer_sizes_q,
            activation=activation,
        )

        self.q1_targ = deepcopy(self.q1)
        self.q2_targ = deepcopy(self.q2)

        # Freeze targets so they are not updated by optimizers
        for p in self.q1_targ.parameters():
            p.requires_grad = False
        for p in self.q2_targ.parameters():
            p.requires_grad = False

        self.buffer = EpisodeBuffer(self.obs_dim, self.act_dim, max_ep_len, max_buf_len)
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.polyak = polyak
        self.gamma = gamma
        self.alpha = alpha
        self.log_alpha = torch.tensor(np.log(self.alpha), requires_grad=True)
        self.update_alpha_after = update_alpha_after
        self.target_entropy = target_entropy
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        self.q_params = chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=q_lr)
        self.alpha_optimizer = Adam([self.log_alpha], lr=a_lr)

        # Create parameter objects for convenience when updating targets
        self.main_params = chain(
            self.pi.parameters(), self.q1.parameters(), self.q2.parameters()
        )
        self.targ_params = chain(self.q1_targ.parameters(), self.q2_targ.parameters())

    def act(self, obs, deterministic=False, **kwargs):
        """Return noisy action as numpy array, **without computing grads**"""
        with torch.no_grad():
            act, _ = self.pi(obs, deterministic=deterministic, get_logprob=False)
            act = act.numpy()
            act = np.squeeze(act, (0, 1))
        return act

    def reset_state(self):
        self.pi.reset_state()
        self.q1.reset_state()
        self.q2.reset_state()

    def store_to_buffer(self, obs, act, rwd, obs_next, done):
        self.buffer.store(obs, act, rwd, obs_next, done)

    def compute_loss_pi(self, data):
        obs = data["obs"]
        act, logprob_pi = self.pi(obs)
        pi_info = dict(LogPi=logprob_pi.detach().numpy())
        q1_pi = self.q1(torch.cat([obs, act], dim=-1))
        q2_pi = self.q2(torch.cat([obs, act], dim=-1))
        q_pi = torch.min(q1_pi, q2_pi)
        logprob_pi = logprob_pi.reshape_as(q_pi)
        loss_pi = (-q_pi + self.alpha * logprob_pi).mean()
        return loss_pi, pi_info

    def compute_loss_q(self, data):
        obs, act, rwd, obs_next, done = (
            data["obs"],
            data["act"],
            data["rwd"],
            data["obs_next"],
            data["done"],
        )

        q1 = self.q1(torch.cat([obs, act], dim=-1))
        q2 = self.q2(torch.cat([obs, act], dim=-1))
        q1 = q1.reshape_as(done)
        q2 = q2.reshape_as(done)

        # Bellman backup for Q function
        with torch.no_grad():
            obs_padded = F.pad(obs, (0, 0, 0, 0, 0, 1), "constant", 0)
            act_padded, logprob_padded = self.pi(obs_padded)

            q1_target = self.q1_targ(torch.cat([obs_padded, act_padded], dim=-1))
            q2_target = self.q2_targ(torch.cat([obs_padded, act_padded], dim=-1))
            q_target_padded = torch.min(q1_target, q2_target)

            # HACK - probably a way to do this in one line with slice?
            if len(q_target_padded.shape) == 3:
                q_target = q_target_padded[1:, :, :]
            elif len(q_target_padded.shape) == 2:
                q_target = q_target_padded[1:, :]
            else:
                raise ValueError("Q tensor has unexpected number of dimensions!")
            logprob_next = logprob_padded[1:, :]

            # reshape so that this will work with SACAgent (for testing)
            q_target = q_target.reshape_as(done)
            logprob_next = logprob_next.reshape_as(done)
            backup = rwd + self.gamma * (1 - done) * (
                q_target - self.alpha * logprob_next
            )

        # MSE loss against Bellman backup
        loss_q = ((q1 - backup) ** 2).mean() + ((q2 - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy(),)

        return loss_q, loss_info

    def update(self, batch_size, t_total):
        # Get training data from buffer
        data = self.buffer.sample_episodes(batch_size)

        # Update Q function
        self.q_optimizer.zero_grad()
        loss_q, loss_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # logger.store(LossQ=loss_q.item(), **loss_info)

        for p in self.q_params:
            p.requires_grad = False

        # Update policy
        self.pi_optimizer.zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q params after policy update
        for p in self.q_params:
            p.requires_grad = True

        # logger.store(LossPi=loss_pi.item(), **pi_info)

        with torch.no_grad():
            for p, p_target in zip(self.q_params, self.targ_params):
                p_target.data.mul_(self.polyak)
                p_target.data.add_((1 - self.polyak) * p.data)

        # Update alpha
        if t_total >= self.update_alpha_after:
            obs = data["obs"]
            pi, log_pi = self.pi(obs)
            loss_alpha = (
                self.log_alpha * (-log_pi - self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
