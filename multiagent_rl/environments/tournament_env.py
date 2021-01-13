from copy import copy
import numpy as np
import pandas as pd
import random
from random import sample
from scipy.stats import rankdata
import torch

import gym
from gym import spaces

EPS = 1e-8


class OneHot(gym.Space):
    """
    One-hot space. Used as the observation space.
    """

    def __init__(self, n):
        super(OneHot, self).__init__()
        self.n = n

    def sample(self):
        return np.random.multinomial(1, [1.0 / self.n] * self.n)

    def contains(self, x):
        return (
            isinstance(x, np.ndarray)
            and x.shape == (self.n,)
            and np.all(np.logical_or(x == 0, x == 1))
            and np.sum(x) == 1
        )

    @property
    def shape(self):
        return (self.n,)

    def __repr__(self):
        return "OneHot(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n


class ConstantDualUltimatum(gym.Env):
    """A single-agent environment consisting of a 'dual ultimatum' game against a constant bot"""

    def __init__(self, ep_len=10, reward="ultimatum", opponent_offer=None, opponent_demand=None, fixed=False):
        super(ConstantDualUltimatum, self).__init__()
        self.fixed = fixed
        if opponent_offer is None:
            self.opponent_offer = np.random.rand()
        else:
            self.opponent_offer = opponent_offer
        if opponent_demand is None:
            self.opponent_demand = np.random.rand()
        else:
            self.opponent_demand = opponent_demand
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        # Can we treat full observation space as contin box, incl first turn flag?
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )
        # self.observation_space = spaces.Tuple(
        #     (
        #         spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
        #         # spaces.Discrete(2),
        #     )
        # )
        self.ep_len = ep_len
        self.current_turn=0
        if reward == "l2":
            self.reward = self._l2_reward
        elif reward == "l1":
            self.reward = self._l1_reward
        elif reward == "l1_const":
            self.reward = self._l1_const_reward
        else:
            self.reward = self._ultimatum_reward

    def _ultimatum_reward(self, action):
        offer, demand = action
        if (
            offer + EPS >= self.opponent_demand
            and self.opponent_offer + EPS >= demand
        ):
            # reward = (1 - offer) + self.opponent_offer
            reward = 1-offer
        else:
            reward = 0
        return reward

    def _l1_reward(self, action):
        """Simple reward for testing"""
        offer_0, _ = action
        l1 = -np.abs(offer_0 - self.opponent_demand)
        reward = np.array([l1, l1])
        return reward

    def _l2_reward(self, action):
        """Simple reward for testing"""
        offer_0, _ = action[0, :]
        l2 = -((offer_0 - self.opponent_demand) ** 2)
        reward = np.array([l2, l2])
        return reward

    def _l1_const_reward(self, action, target=0.3):
        """Simple reward for testing"""
        offer_0, _ = action[0, :]
        r_0 = -np.abs(offer_0 - target)
        reward = r_0
        return reward

    def step(self, action):
        reward = self.reward(action)
        # Add flag indicating this is not the first step
        obs = np.array(
            [self.opponent_offer, self.opponent_demand, action[0], action[1], 0]
        )
        done = 0
        if self.current_turn == (self.ep_len - 1):
            done = 1
            self.reset()
        else:
            done = 0
            self.current_turn += 1
        return obs, reward, done, {}

    def reset(self):
        self.current_turn = 0
        if not self.fixed:
            self.opponent_offer = np.random.rand()
            self.opponent_demand = np.random.rand()
        # Create init state obs, with flag indicating this is the first step
        obs = np.array(
            [self.opponent_offer, self.opponent_demand, 0, 0, 1]
        )
        return obs

    def render(self, mode="human"):
        pass


class StaticDualUltimatum(gym.Env):
    """A single-agent environment consisting of a 'dual ultimatum' game against a StaticDistribBot"""

    def __init__(self, reward="ultimatum", opponent_offer=0.5, opponent_demand=0.5, fixed=False):
        super(StaticDualUltimatum, self).__init__()
        self.fixed = fixed
        self.opponent_offer = opponent_offer
        self.opponent_demand = opponent_demand
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Tuple(
            (
                spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                spaces.Discrete(2),
            )
        )
        if reward == "l2":
            self.rewards = self._l2_rewards
        elif reward == "l1":
            self.rewards = self._l1_rewards
        elif reward == "l1_const":
            self.rewards = self._l1_const_rewards
        else:
            self.rewards = self._ultimatum_rewards

    def _ultimatum_rewards(self, action):
        offer, demand = action
        if (
            offer + EPS >= self.opponent_demand
            and self.opponent_offer + EPS >= demand
        ):
            reward = (1 - offer) + self.opponent_offer
        else:
            reward = 0
        return reward

    def _l1_rewards(self, action):
        """Simple reward for testing"""
        offer_0, _ = action
        l1 = -np.abs(offer_0 - offer_1)
        rewards = np.array([l1, l1])
        return rewards

    def _l2_rewards(self, actions):
        """Simple reward for testing"""
        offer_0, _ = actions[0, :]
        offer_1, _ = actions[1, :]
        l2 = -((offer_0 - offer_1) ** 2)
        rewards = np.array([l2, l2])
        return rewards

    def _l1_const_rewards(self, actions, target=0.3):
        """Simple reward for testing"""
        offer_0, _ = actions[0, :]
        offer_1, _ = actions[1, :]
        r_0 = -np.abs(offer_0 - target)
        r_1 = -np.abs(offer_1 - target)
        rewards = np.array([r_0, r_1])
        return rewards

    def step(self, actions):
        rewards = self.rewards(actions)
        obs = np.array(
            [
                np.concatenate((actions[0, :], actions[1, :])),
                np.concatenate((actions[1, :], actions[0, :])),
            ]
        )
        # Add flag indicating this is not the first step
        obs = np.concatenate((obs, np.zeros((2, 1))), axis=1)
        done = False
        return obs, rewards, done, {}

    def reset(self):
        # Create init state obs, with flag indicating this is the first step
        obs = np.zeros((2, 5))
        obs[:, -1] = 1
        return obs

    def render(self, mode="human"):
        pass


class DualUltimatum(gym.Env):
    """A two-agent environment consisting of a 'dual ultimatum' game'"""

    def __init__(self, reward="ultimatum"):
        super(DualUltimatum, self).__init__()
        self.action_space = spaces.Tuple(
            [
                spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
                for _ in range(2)
            ]
        )
        self.observation_space = spaces.Tuple(
            [
                spaces.Tuple(
                    (
                        spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32),
                        spaces.Discrete(2),
                    )
                )
                for _ in range(2)
            ]
        )
        if reward == "l2":
            self.rewards = self._l2_rewards
        elif reward == "l1":
            self.rewards = self._l1_rewards
        elif reward == "l1_const":
            self.rewards = self._l1_const_rewards
        else:
            self.rewards = self._ultimatum_rewards

    def _ultimatum_rewards(self, actions):
        offer_0, demand_0 = actions[0, :]
        offer_1, demand_1 = actions[1, :]

        if offer_0 + EPS >= demand_1 and offer_1 + EPS >= demand_0:
            reward_0 = (1 - offer_0) + offer_1
            reward_1 = offer_0 + (1 - offer_1)
            rewards = np.array([reward_0, reward_1])
        else:
            rewards = np.array([0, 0])
        return rewards

    def _l1_rewards(self, actions):
        """Simple reward for testing"""
        offer_0, _ = actions[0, :]
        offer_1, _ = actions[1, :]
        l1 = -np.abs(offer_0 - offer_1)
        rewards = np.array([l1, l1])
        return rewards

    def _l2_rewards(self, actions):
        """Simple reward for testing"""
        offer_0, _ = actions[0, :]
        offer_1, _ = actions[1, :]
        l2 = -((offer_0 - offer_1) ** 2)
        rewards = np.array([l2, l2])
        return rewards

    def _l1_const_rewards(self, actions, target=0.3):
        """Simple reward for testing"""
        offer_0, _ = actions[0, :]
        offer_1, _ = actions[1, :]
        r_0 = -np.abs(offer_0 - target)
        r_1 = -np.abs(offer_1 - target)
        rewards = np.array([r_0, r_1])
        return rewards

    def step(self, actions):
        rewards = self.rewards(actions)
        obs = np.array(
            [
                np.concatenate((actions[0, :], actions[1, :])),
                np.concatenate((actions[1, :], actions[0, :])),
            ]
        )
        # Add flag indicating this is not the first step
        obs = np.concatenate((obs, np.zeros((2, 1))), axis=1)
        done = False
        return obs, rewards, done, {}

    def reset(self):
        # Create init state obs, with flag indicating this is the first step
        obs = np.zeros((2, 5))
        obs[:, -1] = 1
        return obs

    def render(self, mode="human"):
        pass


class MatrixGame(gym.Env):
    """An environment consisting of a matrix game with stochastic outcomes"""

    NUM_ACTIONS = 3
    NUM_STATES = NUM_ACTIONS ** 2 + 1

    def __init__(
        self,
        payout_mean=np.array([[10, 5, -5], [0, 0, 5], [20, -5, 0]]),
        payout_std=np.array([[0, 0, 0], [0, 0, 0], [0, 20, 20]]),
    ):
        super(MatrixGame, self).__init__()
        self.num_actions = 3
        self.action_space = spaces.Tuple(
            [spaces.Discrete(self.num_actions) for _ in range(2)]
        )
        self.observation_space = spaces.Tuple(
            [OneHot(self.NUM_STATES) for _ in range(2)]
        )
        self.payout_mean_matrix = payout_mean
        self.payout_std_matrix = payout_std

    def step(self, actions):
        a0, a1 = actions

        reward = np.array(
            [
                self.payout_mean_matrix[a0, a1]
                + self.payout_std_matrix[a0, a1] * np.random.randn(1)[0],
                self.payout_mean_matrix[a1, a0]
                + self.payout_std_matrix[a1, a0] * np.random.randn(1)[0],
            ]
        )

        obs0 = np.zeros(self.NUM_STATES)
        obs1 = np.zeros(self.NUM_STATES)
        obs0[a0 * self.NUM_ACTIONS + a1] = 1
        obs1[a1 * self.NUM_ACTIONS + a0] = 1
        obs = np.array([obs0, obs1])
        done = False
        return obs, reward, done, {}

    def reset(self):
        init_state = np.zeros(self.NUM_STATES)
        init_state[-1] = 1
        obs = np.array([init_state, init_state])
        return obs

    def render(self, mode="human"):
        pass


def assign_match_pairs(num_agents):
    """Create random pairings for an even number of agents"""
    assert num_agents % 2 == 0
    shuffled = sample(range(num_agents), k=num_agents)
    match_pairs = [shuffled[i : i + 2] for i in range(0, num_agents, 2)]
    return match_pairs


class RoundRobinTournament(gym.Env):
    """An environment for of a tournament of some pairwise game."""

    def __init__(
        self,
        num_agents,
        num_rounds=10,
        round_length=10,
        noise_size=1,
        top_cutoff=2,
        bottom_cutoff=1,
        top_reward=1.0,
        bottom_reward=1.0,
        game_fn=DualUltimatum,
        score_reward=False,
        per_turn_reward=False,
        center_scores=False,
        hide_obs=False,
        hide_score=True,
        game_kwargs=dict(),
    ):
        super(RoundRobinTournament, self).__init__()

        self.num_agents = num_agents
        self.num_matches = int(num_agents / 2)
        self.num_rounds = num_rounds
        self.round_length = round_length
        self.noise_size = noise_size
        self.top_cutoff = top_cutoff
        self.bottom_cutoff = bottom_cutoff
        self.top_reward = top_reward
        self.bottom_reward = bottom_reward
        self.score_reward = score_reward
        self.per_turn_reward = per_turn_reward
        self.center_scores = center_scores
        self.hide_obs = hide_obs
        self.hide_score = hide_score
        self.match_env_list = [game_fn(**game_kwargs)] * int(num_agents / 2)
        self.action_space = spaces.Tuple(
            [env.action_space for env in self.match_env_list]
        )

        # hard coded for now (to dualultimatum)
        # observation space for single match
        self.match_obs_dim = 5
        # observation space for tournament
        self.obs_dim = self.match_obs_dim + num_agents + 1 + 1
        # Observations are:
        # 4 (match obs), contin
        # 1 first turn flag, binary
        # 1 rounds left, int
        # 1 opponent score, contin
        # num_agents (scores), contin
        # ? should we add more for rankings/thresholds?
        # opponent id, OHE??? useful when we add bots, or other differences?

        # Create tuple of observations for all agents. Extend match-level obs with tournament-level obs features
        self.observation_space = spaces.Tuple(
            [
                spaces.Tuple(
                    [space for space in self.match_env_list[0].observation_space[0]]
                    + [
                        spaces.Discrete(self.num_rounds),
                        spaces.Box(
                            low=0.0,
                            high=np.inf,
                            shape=(self.num_agents + 1,),
                            dtype=np.float32,
                        ),
                    ]
                )
                for _ in range(self.num_agents)
            ]
        )

    def _final_reward(self):
        if self.score_reward:
            return self.scores
        else:
            # print(f'final scores {self.scores}')
            ranks = rankdata(-self.scores)
            reward = (ranks <= self.top_cutoff) * self.top_reward
            if self.bottom_cutoff is not None:
                ranks = rankdata(+self.scores)
                reward += (ranks <= self.bottom_cutoff) * self.bottom_reward
            return reward

    def step(self, actions):
        # Rearrange actions as input for each match environment
        match_actions = [
            np.array([actions[i] for i in match]) for match in self.match_pairs
        ]

        # Pass actions to each match env, get next obs/reward
        match_outputs = [
            match_env.step(acts)
            for acts, match_env in zip(match_actions, self.match_env_list)
        ]

        all_obs = np.zeros((self.num_agents, self.obs_dim))
        # Update scores/obs based on env steps
        for pair, output in zip(self.match_pairs, match_outputs):
            o, r, d, _ = output
            self.scores[pair] += r
            # obs - match, last moves
            all_obs[pair[0], : self.match_obs_dim] = o[0]
            all_obs[pair[1], : self.match_obs_dim] = o[1]

        if self.center_scores:
            self.scores -= np.mean(self.scores)
        # obs - current round / rounds left
        all_obs[:, self.match_obs_dim] = self.current_round / self.num_rounds
        # obs - opponent score
        all_obs[:, self.match_obs_dim + 1] = [
            self.scores[i] for i in self.agent_opponent
        ]
        # obs - all agents' scores
        all_obs[:, self.match_obs_dim + 2 :] = self.scores

        reward = np.zeros(self.num_agents)
        done = False
        self.current_turn -= 1

        # Per turn reward based on change to relative score
        if self.per_turn_reward:
            reward = np.zeros(self.num_agents)
            for pair, output in zip(self.match_pairs, match_outputs):
                _, r, _, _ = output
                reward[pair] = r - np.sum(r) / self.num_agents
                # raw score reward
                # reward[pair] = r
        # end of episode return
        elif (self.current_turn == 0) and (self.current_round == 1):
            reward = self._final_reward()

        if (self.current_turn == 0) and (self.current_round == 1):
            done = True

        # start next round
        if self.current_turn == 0:
            self.match_pairs = assign_match_pairs(self.num_agents)
            self.current_round -= 1
            self.current_turn = self.round_length
            # to do: check that this actually resets state for match environments
            all_obs[:, : self.match_obs_dim] = np.concatenate(
                [match.reset() for match in self.match_env_list]
            )

        if self.hide_obs:
            # add a hack to keep own score as feature (for delayed reward)
            if self.hide_score:
                all_obs = 0.0 * all_obs
            else:
                keep_col = self.match_obs_dim + 2
                all_obs[:, :keep_col] = 0
                # all_obs[:, keep_col + 1:] = 0
        return all_obs, reward, done, {}

    def reset(self):
        self.current_round = self.num_rounds
        self.current_turn = self.round_length
        self.scores = np.random.randn(self.num_agents) * self.noise_size
        self.match_pairs = assign_match_pairs(self.num_agents)
        self.agent_opponent = np.zeros(self.num_agents, dtype=np.int32)
        self.agent_match = np.zeros(self.num_agents, dtype=np.int32)
        for i, m in enumerate(self.match_pairs):
            # Note each agent's opponent
            self.agent_opponent[m[0]] = m[1]
            self.agent_opponent[m[1]] = m[0]
            # Note each agent's match
            self.agent_match[m[0]] = i
            self.agent_match[m[1]] = i

        # Generate initial obs for each match
        all_obs = np.zeros((self.num_agents, self.obs_dim))

        match_obs = [match_env.reset() for match_env in self.match_env_list]

        for i in range(self.num_matches):
            p0 = self.match_pairs[i][0]
            p1 = self.match_pairs[i][1]
            all_obs[p0, : self.match_obs_dim] = match_obs[i][0]
            all_obs[p1, : self.match_obs_dim] = match_obs[i][1]
        # obs - current round / rounds left
        all_obs[:, self.match_obs_dim] = self.current_round / self.num_rounds
        # obs - opponent score
        all_obs[:, self.match_obs_dim + 1] = [
            self.scores[i] for i in self.agent_opponent
        ]
        # obs - all agents' scores
        all_obs[:, self.match_obs_dim + 2 :] = self.scores

        if self.hide_obs:
            all_obs = 0.0 * all_obs
        return all_obs

    def render(self, mode="human"):
        pass

    def get_state(self):
        """
        Return full state of environment for logging.
        """

        env_state = dict(
            round=self.current_round,
            turn=self.current_turn,
            opponent=copy(self.agent_opponent),
            score=copy(np.round(self.scores, 3)),
        )

        # env_state = pd.DataFrame(
        #     dict(
        #         round=[self.current_round],
        #         turn=[self.current_turn],
        #         opponent=[copy(self.agent_opponent)],
        #         scores=[copy(np.round(self.scores, 3))],
        #     )
        # ).copy(deep=True)

        return env_state


class MimicObs(gym.Env):
    """A simple environment for testing in which reward is based on distance to last obs"""

    def __init__(self, ep_len=10, reward="l1", target="last", goal_constant=None, goal_mean=False):
        super(MimicObs, self).__init__()
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.ep_len = ep_len
        self.current_turn = 0
        self.running_sum = np.zeros(1)
        self.target = target
        self.goal_constant = goal_constant
        self.goal_mean = goal_mean
        self.last_obs = np.random.rand(1)
        if reward == "l2":
            self.reward = lambda a, b: -((a - b) ** 2)
        elif reward == "l1":
            self.reward = lambda a, b: -np.abs(a - b)

    def step(self, action):
        if self.goal_constant is not None:
            target = self.goal_constant
        elif self.goal_mean:
            self.running_sum += self.last_obs
            target = self.running_sum / (self.current_turn + 1)
        else:
            target = self.last_obs

        reward = self.reward(action, target)
        self.last_obs = np.random.rand(1)
        obs = self.last_obs
        if self.current_turn == (self.ep_len - 1):
            done = 1
            self.current_turn = 0
            self.running_sum = np.zeros(1)
        else:
            done = 0
            self.current_turn += 1
        return obs, reward, done, {}

    def reset(self):
        self.current_turn = 0
        self.last_obs = np.random.rand(1)
        obs = self.last_obs
        return obs

    def render(self, mode="human"):
        pass
