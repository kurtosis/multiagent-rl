import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys

sys.path.insert(
    0, "/Users/kurtsmith/research/pytorch_projects/reinforcement_learning/environments"
)
sys.path.insert(0, "/Users/kurtsmith/research/spinningup")

from copy import deepcopy
import numpy as np
import time

from gym.spaces import Box, Discrete, Tuple
import torch
from torch.distributions import Normal
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

from multiagent_rl.algos.agents import *
from multiagent_rl.algos.buffers import *
from multiagent_rl.algos.training import *
from multiagent_rl.environments.tournament_env import *
from multiagent_rl.utils.logx import EpochLogger


def mlp(layer_sizes, hidden_activation, final_activation):
    layers = []
    for i in range(len(layer_sizes) - 1):
        layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i < len(layer_sizes) - 2:
            layers.append(hidden_activation())
        else:
            layers.append(final_activation())
    return nn.Sequential(*layers)


class ConstantBot:
    """
    Static bot that plays a constant action in Dual Ultimatum game.
    """

    def __init__(
        self,
        *args,
        offer=None,
        demand=None,
        mean_offer=None,
        std_offer=None,
        mean_demand=None,
        std_demand=None,
        **kwargs,
    ):
        def set_action(value, mean, std):
            if value is not None:
                return value
            elif mean is not None and std is not None:
                return (1 + np.tanh(mean_offer + std_offer * np.random.randn(1)[0])) / 2
            else:
                return np.random.rand(1)[0]

        self.offer = set_action(offer, mean_offer, std_offer)
        self.demand = set_action(demand, mean_demand, std_demand)

    def act(self, *args, **kwargs):
        return np.array((self.offer, self.demand))

    def update(self, *args, **kwargs):
        pass


class StaticDistribBot:
    """
    Bot that plays a draw from a static distribution, based on tanh transform.
    To do: Could implement this using beta or log-odds normal distr instead, easier to reason about?
    """

    def __init__(
        self,
        *args,
        mean_offer=0.5,
        std_offer=1.0,
        mean_demand=0.5,
        std_demand=1.0,
        **kwargs,
    ):
        # Initialized with approximate mean values (in (0,1)) for simplicity.
        # Note these aren't exact means b/c of the nonlinear tanh transform.
        self.approx_mean_offer = mean_offer
        self.mean_tanh_offer = np.arctanh(2 * mean_offer - 1)
        self.std_offer = std_offer
        self.approx_mean_demand = mean_demand
        self.mean_tanh_demand = np.arctanh(2 * mean_demand - 1)
        self.std_demand = std_demand

    def act(self):
        offer = (
            1 + np.tanh(self.mean_tanh_offer + self.std_offer * np.random.randn(1)[0])
        ) / 2
        demand = (
            1 + np.tanh(self.mean_tanh_demand + self.std_demand * np.random.randn(1)[0])
        ) / 2
        return np.array((offer, demand))

    def update(self, *args, **kwargs):
        pass


class MimicBot:
    def __init__(self):
        pass

    def act(self, last_offer=None, last_demand=None):
        if last_offer is None:
            last_offer = np.random.rand(1)
        if last_demand is None:
            last_demand = np.random.rand(1)
        return np.array((last_offer, last_demand))


class BenchmarkBot:
    def __init__(self, benchmark=1):
        self.benchmark = benchmark
        self.cum_total = 0


class GreedFearBot:
    def __init__(self, greed=1, fear=1):
        self.greed = greed
        self.fear = fear

    def act(self, last_offer, last_demand, accepted):
        if accepted:
            offer = last_offer - 0
            demand = last_demand + 0
        else:
            offer = last_offer + 0
            demand = last_demand - 0
        return np.array((offer, demand))


class DualUltimatumSACAgent(nn.Module):
    """
    Agent to be used in SAC for Dual Ultimatum.
    (Is this the same as a normal SAC agent?)
    Contains:
    - stochastic policy (bounded by tanh)
    - estimated Q*(s,a,)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_layers_pi=(64, 64),
        hidden_layers_q=(64, 64),
        activation=nn.ReLU,
        **kwargs,
    ):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        layer_sizes_pi = [obs_dim] + hidden_layers_pi
        layer_sizes_q = [obs_dim + act_dim] + hidden_layers_q + [1]
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
            act, _ = self.policy(obs, deterministic=deterministic)
        return act.numpy()


def dualultimatum_bots(
    player_1,
    player_2,
    env_fn=DualUltimatum,
    seed=0,
    epochs=100,
    steps_per_epoch=4000,
    max_episode_len=1000,
    logger_kwargs=dict(),
):
    """Run 'training' between two bots."""
    # Initialize environment, agent, auxiliary objects

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    act_low = env.action_space.low
    act_high = env.action_space.high

    # Set up model saving
    logger.setup_pytorch_saver(player_1)

    start_time = time.time()

    # Begin training phase.
    for epoch in range(epochs):
        obs = env.reset()
        episode_return = np.array([0.0, 0.0])
        episode_length = 0
        for t in range(steps_per_epoch):
            # Take random actions for first n steps to do broad exploration
            act_1 = player_1.act()
            act_2 = player_2.act()
            act = np.concatenate((act_1, act_2))
            # Step environment given latest agent action
            obs_next, reward, done, _ = env.step(act)

            episode_return += reward
            episode_length += 1

            # check if episode is over
            episode_capped = episode_length == max_episode_len
            epoch_ended = t == steps_per_epoch - 1
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                if done or episode_capped:
                    logger.store(
                        EpRet1=episode_return[0],
                        EpRet2=episode_return[1],
                        EpLen=episode_length,
                    )
                obs = env.reset()
                episode_return = np.array([0.0, 0.0])
                episode_length = 0

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet1", with_min_and_max=True)
        logger.log_tabular("EpRet2", with_min_and_max=True)
        # logger.log_tabular("TestEpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        # logger.log_tabular("TestEpLen", average_only=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        # logger.log_tabular("Q1Vals", with_min_and_max=True)
        # logger.log_tabular("Q2Vals", with_min_and_max=True)
        # logger.log_tabular("LogPi", with_min_and_max=True)
        # logger.log_tabular("LossPi", average_only=True)
        # logger.log_tabular("LossQ1", average_only=True)
        # logger.log_tabular("LossQ2", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()


def dualultimatum_ddpg(
    agent_1_fn=DDPGAgent,
    agent_1_kwargs=dict(),
    agent_2_fn=DDPGAgent,
    agent_2_kwargs=dict(),
    # player_2=ConstantBot,
    env_fn=DualUltimatum,
    seed=0,
    epochs=100,
    steps_per_epoch=4000,
    replay_size=1000000,
    sample_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    test_episodes=10,
    log_interval=10,
    max_episode_len=1000,
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    logger_kwargs=dict(),
    save_freq=10,
):
    """Run DDPG training."""
    # Initialize environment, agent, auxiliary objects

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = env_fn()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    agent_1 = agent_1_fn(env.observation_space, env.action_space, **agent_1_kwargs)
    agent_2 = agent_2_fn(env.observation_space, env.action_space, **agent_2_kwargs)

    buf = TransitionBuffer(obs_dim, act_dim, replay_size)

    multi_buf = MultiagentTransitionBuffer(obs_dim, act_dim, 2, replay_size)

    # Set up model saving
    logger.setup_pytorch_saver(agent_1)

    def deterministic_policy_test():
        for _ in range(test_episodes):
            o = test_env.reset()
            ep_ret = np.array([0.0, 0.0])
            ep_len = 0
            d = False
            while not d and not ep_len == max_episode_len:
                with torch.no_grad():
                    a1 = agent_1.act(
                        torch.as_tensor(o, dtype=torch.float32), noise=False
                    )
                    a2 = agent_2.act(
                        torch.as_tensor(o, dtype=torch.float32), noise=False
                    )
                a = np.stack((a1, a2))
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(
                TestEpRet1=ep_ret[0], TestEpRet2=ep_ret[1], TestEpLen=ep_len,
            )

    start_time = time.time()

    # Begin training phase.
    t_total = 0
    update_time = 0.0
    for epoch in range(epochs):
        obs = env.reset()
        episode_return = np.array([0.0, 0.0])
        episode_length = 0
        for t in range(steps_per_epoch):

            act_1 = agent_1.act(torch.as_tensor(obs, dtype=torch.float32), noise=True)
            act_2 = agent_2.act(torch.as_tensor(obs, dtype=torch.float32), noise=True)
            # act = np.concatenate((act_1, act_2))

            act = np.stack((act_1, act_2))
            obs_next, reward, done, _ = env.step(act)

            episode_return += reward
            episode_length += 1

            # Store current step in buffer
            # buf.store(obs, act_1, rwd[0], obs_next, done)
            # buf.store(obs, act_2, rwd[1], obs_next, done)
            multi_buf.store(obs, act, reward, obs_next, done)

            # update episode return and env state
            obs = obs_next

            # check if episode is over
            episode_capped = episode_length == max_episode_len
            epoch_ended = t == steps_per_epoch - 1
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                if done or episode_capped:
                    logger.store(
                        EpRet1=episode_return[0],
                        EpRet2=episode_return[1],
                        EpLen=episode_length,
                    )
                obs = env.reset()
                episode_return = np.array([0.0, 0.0])
                episode_length = 0

            if t_total >= update_after and (t + 1) % update_every == 0:
                for _ in range(update_every):
                    data = multi_buf.get(sample_size=sample_size)
                    agent_1.update(data, agent=0)
                    agent_2.update(data, agent=1)

            t_total += 1

        deterministic_policy_test()

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet1", with_min_and_max=True)
        logger.log_tabular("TestEpRet1", with_min_and_max=True)
        logger.log_tabular("EpRet2", with_min_and_max=True)
        logger.log_tabular("TestEpRet2", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("TestEpLen", average_only=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        # logger.log_tabular("QVals", with_min_and_max=True)
        # logger.log_tabular("LossPi", average_only=True)
        # logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()


def dualultimatum_td3(
    agent_fn=TD3Agent,
    player_2=ConstantBot,
    env_fn=DualUltimatum,
    seed=0,
    epochs=100,
    steps_per_epoch=4000,
    replay_size=1000000,
    sample_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    test_episodes=10,
    log_interval=10,
    max_episode_len=1000,
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    agent_kwargs=dict(),
    logger_kwargs=dict(),
    save_freq=10,
    policy_delay=2,
    target_noise_std=0.2,
    target_clip=0.5,
):
    """Run TD3 training."""
    # Initialize environment, agent, auxiliary objects

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = env_fn()
    # env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    act_low = env.action_space.low
    act_high = env.action_space.high
    agent = agent_fn(env.observation_space, env.action_space, **agent_kwargs)
    agent_target = deepcopy(agent)

    # Freeze target mu, Q so they are not updated by optimizers
    for p in agent_target.parameters():
        p.requires_grad = False

    var_counts = tuple(count_vars(module) for module in [agent.pi, agent.q1])
    logger.log(
        f"\nNumber of parameters \t policy: {var_counts[0]} q1/q2: {var_counts[1]}\n"
    )

    buf = TransitionBuffer(obs_dim, act_dim, replay_size)
    pi_optimizer = Adam(agent.pi.parameters(), lr=pi_lr)
    q1_optimizer = Adam(agent.q1.parameters(), lr=q_lr)
    q2_optimizer = Adam(agent.q2.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(agent)

    def compute_loss_policy(data):
        # get data
        o = data["obs"]
        # Get actions that agent policy would take at each step
        a = agent.pi(o)
        return -agent.q1(torch.cat((o, a), dim=-1)).mean()

    def compute_q_target(data):
        r, o_next, d = data["rwd"], data["obs_next"], data["done"]
        with torch.no_grad():
            a_next = agent_target.pi(o_next)
            noise = np.random.randn(*a_next.shape) * target_noise_std
            noise = np.clip(noise, -target_clip, +target_clip)
            a_next += noise.astype("float32")
            a_next = np.clip(a_next, act_low, act_high)
            q1_target = agent_target.q1(torch.cat((o_next, a_next), dim=-1))
            q2_target = agent_target.q2(torch.cat((o_next, a_next), dim=-1))
            q_target = torch.min(q1_target, q2_target)
            q_target = r + gamma * (1 - d) * q_target
        return q_target

    def compute_loss_q(q_model, data, q_target, qvals="QVals"):
        o, a = data["obs"], data["act"]
        q = q_model(torch.cat((o, a), dim=-1))
        q_loss_info = {qvals: q.detach().numpy()}
        return ((q - q_target) ** 2).mean(), q_loss_info

    def update(i):
        # Get training data from buffer
        data = buf.sample_batch()

        # Update Q function
        q1_optimizer.zero_grad()
        q2_optimizer.zero_grad()
        q_target = compute_q_target(data)
        q1_loss, q1_loss_info = compute_loss_q(agent.q1, data, q_target, qvals="Q1Vals")
        q2_loss, q2_loss_info = compute_loss_q(agent.q2, data, q_target, qvals="Q2Vals")
        q1_loss.backward()
        q1_optimizer.step()
        q2_loss.backward()
        q2_optimizer.step()

        logger.store(
            LossQ1=q1_loss.item(),
            LossQ2=q2_loss.item(),
            **q1_loss_info,
            **q2_loss_info,
        )

        if i % policy_delay == 0:
            # Freeze Q params during policy update to save time
            for p in agent.q1.parameters():
                p.requires_grad = False
            for p in agent.q2.parameters():
                p.requires_grad = False
            # Update policy
            pi_optimizer.zero_grad()
            pi_loss = compute_loss_policy(data)
            pi_loss.backward()
            pi_optimizer.step()
            # Unfreeze Q params after policy update
            for p in agent.q1.parameters():
                p.requires_grad = True
            for p in agent.q2.parameters():
                p.requires_grad = True

            with torch.no_grad():
                for p, p_targ in zip(
                    agent_target.parameters(), agent_target.parameters()
                ):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

            logger.store(LossPi=pi_loss.item(),)

    def deterministic_policy_test():
        for _ in range(test_episodes):
            o = test_env.reset()
            ep_ret = 0
            ep_len = 0
            d = False
            while not d and not ep_len == max_episode_len:
                with torch.no_grad():
                    a = agent.act(torch.as_tensor(o, dtype=torch.float32), noise=False)
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()

    # Begin training phase.
    t_total = 0
    for epoch in range(epochs):
        obs = env.reset()
        episode_return = np.array([0.0, 0.0])
        episode_length = 0
        for t in range(steps_per_epoch):
            # Take random actions for first n steps to do broad exploration
            if t_total < start_steps:
                act_1 = env.action_space.sample()
            else:
                act_1 = agent.act(torch.as_tensor(obs, dtype=torch.float32), noise=True)
            # Step environment given latest agent action

            act_2 = player_2.act()
            act = np.concatenate((act_1, act_2))

            obs_next, reward, done, _ = env.step(act)

            episode_return += reward
            episode_length += 1

            # Store current step in buffer
            buf.store(obs, act_1, reward[0], obs_next, done)

            # update episode return and env state
            obs = obs_next

            # check if episode is over
            episode_capped = episode_length == max_episode_len
            epoch_ended = t == steps_per_epoch - 1
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                if done or episode_capped:
                    logger.store(
                        EpRet1=episode_return[0],
                        EpRet2=episode_return[1],
                        EpLen=episode_length,
                    )
                obs = env.reset()
                episode_return = np.array([0.0, 0.0])
                episode_length = 0

            if t_total >= update_after and (t + 1) % update_every == 0:
                # update_start = time.time()
                for i_update in range(update_every):
                    # update(i_update)
                    pass
                # update_end = time.time()
                # print(f'update time {update_end - update_start}')

            t_total += 1

        # deterministic_policy_test()
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet1", with_min_and_max=True)
        logger.log_tabular("EpRet2", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
