"""
Dual Ultimatum Environments
This is a series of environments based on the Ultimatum game.
(https://en.wikipedia.org/wiki/Ultimatum_game)
All environments satisfy the OpenAI Gym API.
These environments are based on a modified "Dual Ultimatum" game between two agents. A single turn
of this game can be thought of as follows:
- Each agent is given $1 to share with the other agent. (for $2 total at stake)
- Each agent decides how much of their $1 to "offer" to the other agent. (They keep the remainder.)
- Each agent also decides how much they will "demand" the other agent offer them.
- All offers and demands are revealed simultaneously.
- If both offers are above their corresponding demands, the agents split the $2 per the offer amounts.
- If either offer is below its corresponding demand, both agents receive $0.
The environments below implement several variants of Dual Ultimatum game.
- A basic iterated game for a single learning agent, playing against a bot that does not learn.
- An iterated game for two agents, both of which can potentially be learning agents.
- Tournaments in which four or more agents take turns playing each other. In a tournament environment,
    rewards are based on the final rankings, rather than simply being each agent's cumulative score. (For instance,
    only the first-place finisher, after N rounds, might receive a non-zero reward.)
"""

from copy import copy
import numpy as np
from random import sample
from scipy.stats import rankdata

import gym
from gym import spaces

from multiagent_rl.agents.bot_agents import *


EPS = 1e-8


class ConstantDualUltimatum(gym.Env):
    """A single-agent environment consisting of a dual ultimatum game against a constant bot"""

    def __init__(
        self,
        ep_len=10,
        opponent_offer=None,
        opponent_demand=None,
        fixed=True,
    ):
        super().__init__()
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
        self.ep_len = ep_len
        self.current_turn = 0

    def _reward(self, action):
        offer, demand = action
        if offer + EPS >= self.opponent_demand and self.opponent_offer + EPS >= demand:
            reward = (1 - offer) + self.opponent_offer
        else:
            reward = 0
        return reward

    def step(self, action):
        reward = self._reward(action)
        # Add flag indicating this is not the first step
        obs = np.array(
            [self.opponent_offer, self.opponent_demand, action[0], action[1], 0]
        )
        if self.current_turn == (self.ep_len - 1):
            done = 1
            temp = self.reset()
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
        obs = np.array([self.opponent_offer, self.opponent_demand, 0, 0, 1])
        return obs

    def render(self, mode="human"):
        pass


class DistribDualUltimatum(gym.Env):
    """A single-agent environment consisting of a dual ultimatum game against a DistribBot"""

    def __init__(
        self,
        ep_len=10,
        opponent_offer_mean=None,
        opponent_demand_mean=None,
        opponent_offer_std=None,
        opponent_demand_std=None,
        fixed=False,
    ):
        super().__init__()
        self.ep_len = ep_len
        self.current_turn = 0
        self.fixed = fixed

        # Sample mean from uniform dist
        if opponent_offer_mean is None:
            opponent_offer_mean = np.random.rand()
        if opponent_demand_mean is None:
            opponent_demand_mean = np.random.rand()

        # Sample std from uniform dist
        if opponent_offer_std is None:
            opponent_offer_std = np.random.rand()
        if opponent_demand_std is None:
            opponent_demand_std = np.random.rand()

        self.opponent = DistribBot(
            mean_offer=opponent_offer_mean,
            std_offer=opponent_offer_std,
            mean_demand=opponent_demand_mean,
            std_demand=opponent_demand_std,
        )

        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(5,), dtype=np.float32
        )

    def _reward(self, action, opp_action):
        offer, demand = action
        opponent_offer, opponent_demand = opp_action
        if offer + EPS >= opponent_demand and opponent_offer + EPS >= demand:
            reward = (1 - offer) + opponent_offer
        else:
            reward = 0
        return reward

    def step(self, action):
        opp_action = self.opponent.act()
        reward = self._reward(action, opp_action)
        obs = np.concatenate((opp_action, action, [0]))
        if self.current_turn == (self.ep_len - 1):
            done = 1
            _ = self.reset()
        else:
            done = 0
            self.current_turn += 1

        return obs, reward, done, {}

    def reset(self):
        self.current_turn = 0
        if not self.fixed:
            opponent_offer_mean = np.random.rand()
            opponent_offer_std = np.random.rand()
            opponent_demand_mean = np.random.rand()
            opponent_demand_std = np.random.rand()
            self.opponent = DistribBot(
                mean_offer=opponent_offer_mean,
                std_offer=opponent_offer_std,
                mean_demand=opponent_demand_mean,
                std_demand=opponent_demand_std,
            )
        # Create init state obs, with flag indicating this is the first step
        obs = np.array([0, 0, 0, 0, 1])
        return obs

    def render(self, mode="human"):
        pass


class DualUltimatum(gym.Env):
    """A two-agent environment consisting of a dual ultimatum game between any two agents."""

    def __init__(self, ep_len=10):
        super().__init__()
        self.ep_len = ep_len
        self.current_turn = 0
        self.action_space = spaces.Tuple(
            [
                spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
                for _ in range(2)
            ]
        )
        self.observation_space = spaces.Tuple(
            [
                spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
                for _ in range(2)
            ]
        )

    def _rewards(self, actions):
        offer_0, demand_0 = actions[0]
        offer_1, demand_1 = actions[1]

        if offer_0 + EPS >= demand_1 and offer_1 + EPS >= demand_0:
            reward_0 = (1 - offer_0) + offer_1
            reward_1 = offer_0 + (1 - offer_1)
            rewards = np.array([reward_0, reward_1])
        else:
            rewards = np.array([0, 0])
        return rewards

    def step(self, actions):
        rewards = self._rewards(actions)
        obs = np.array(
            [
                np.concatenate((actions[0], actions[1])),
                np.concatenate((actions[1], actions[0])),
            ]
        )
        # Add flag indicating this is not the first step
        obs = np.concatenate((obs, np.zeros((2, 1))), axis=1)
        if self.current_turn == (self.ep_len - 1):
            done = 1
            _ = self.reset()
        else:
            done = 0
            self.current_turn += 1
        return obs, rewards, done, {}

    def reset(self):
        self.current_turn = 0
        # Create init state obs, with flag indicating this is the first step
        obs = np.zeros((2, 5))
        obs[:, -1] = 1
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
    """An environment for of a Dual Ultimatum tournament."""

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
        super().__init__()

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
        def agent_observation_space():
            return spaces.Tuple(
                [
                    spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32),
                    spaces.Discrete(self.num_rounds),
                    spaces.Box(
                        low=0.0,
                        high=np.inf,
                        shape=(self.num_agents + 1,),
                        dtype=np.float32,
                    ),
                ]
            )

        self.observation_space = spaces.Tuple(
            [agent_observation_space() for _ in range(self.num_agents)]
        )

    def _final_reward(self):
        """Reward based on ranking at end of tournament."""
        if self.score_reward:
            return self.scores
        else:
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
                # raw score rwd
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

        return env_state


class MimicObs(gym.Env):
    """A simple environment for testing in which reward is based on distance to last obs"""

    def __init__(
        self, ep_len=10, reward="l1", target="last", goal_constant=None, goal_mean=False
    ):
        super().__init__()
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
