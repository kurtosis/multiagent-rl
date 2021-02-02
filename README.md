# multiagent-rl

This repository contains several classes for training reinforcement learning agents in a multi-agent
environment based on the [Ultimatum Game](https://en.wikipedia.org/wiki/Ultimatum_game). 
The core training method is based on the [Recurrent Deterministic Policy Gradient](https://arxiv.org/abs/1512.04455)
for LSTM-based agents. However, stochastic agents, based on the 
[Soft Actor-Critic algorithm](https://arxiv.org/abs/1801.01290) 
(see also [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html#))
are used, rather than deterministic agents. A future direction is to incorporate methods specifically
designed for multi-agent environments such as [MADDPG](https://arxiv.org/abs/1706.02275).

In parallel with this code I'm writing a [blog on RL topics](https://kurtsmith.space/blog).

Code is organized in ```./multiagent_rl/``` as follows:
- ```runs/``` The main scripts to run to train agents and evaluate performance. I have not yet modified these to accept command line arguments. 
  For now, arguments should be set directly in the file
- ```agents/``` RL agents as well as bot agents to be used as opponents
- ```buffers.py``` Replay buffers which store data to train agents on
- ```environments.py``` Single-agent and multi-agent environments for the Dual Ultimatum game described below
- ```trainers/``` Training loops. These are the top level functions that can be run to train agents
- ```utils/``` Various utilities, notably EpochLogger which is copied from [Spinning Up](https://github.com/openai/spinningup/tree/master/spinup/utils) as noted
- ```tests/``` Tests


The main motivation for this work is to explore the behavior of RL agents (with varying levels of sophistication)
in a multi-agent environment that has mixed cooperative-competitive dynamics.

The environments are based on modified "dual" version of the
[Ultimatum game](https://en.wikipedia.org/wiki/Ultimatum_game) between two agents.
A single turn
of this game can be thought of as follows:
- Each agent is given $1 to share with the other agent. (for $2 total at stake)
- Each agent decides how much of their $1 to "offer" to the other agent. (They keep the remainder.)
- Each agent also decides how much they will "demand" the other agent offer them.
- All offers and demands are revealed simultaneously.
- If both offers are above their corresponding demands, the agents split the $2 per the offer amounts.
- If either offer is below its corresponding demand, both agents receive $0.

A few variants of this Dual Ultimatum game are implemented:
- A basic iterated game for a single learning agent, playing against a bot that does not learn.
- An iterated game for two agents, one or both of which may potentially be learning agents.
- Tournaments in which four or more agents take turns playing each other. In a tournament environment,
    rewards are based on the final rankings, rather than simply being each agent's cumulative score. (For instance,
    only the first-place finisher, after N rounds, might receive a non-zero reward.)
