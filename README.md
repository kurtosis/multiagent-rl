# multiagent-rl

This repository contains several classes for training reinforcement learning agents in a multiagent 
environment based on the [Ultimatum Game](https://en.wikipedia.org/wiki/Ultimatum_game). 
The core training method is based on the [Recurrent Deterministic Policy Gradient](https://arxiv.org/abs/1512.04455)
for LSTM-based agents. However, stochastic agents, based on the 
[Soft Actor-Critic algorithm](https://arxiv.org/abs/1801.01290) 
(see also [Spinning Up](https://spinningup.openai.com/en/latest/algorithms/sac.html#))
are used, rather than deterministic agents.

Code is organized as follows:
- ./agents/ - RL agents as well as bot agents to be used as opponents.
- ./environments/ - single-agent and multi-agent environments for the Dual Ultimatum game described below.
- ./algos/ - training loops. These are the top level functions that can be run to train agents
- buffers.py - replay buffer which stores data to train agents on.
- ./utils/ various utilities, notably EpochLogger. Much of the contents are copied directly from 
[Spinning Up](https://github.com/openai/spinningup/tree/master/spinup/utils) as noted.
