from itertools import chain
import time

from torch.optim import Adam

from multiagent_rl.environments.tournament_env import *
from multiagent_rl.agents import *
from multiagent_rl.buffers import *
# from multiagent_rl.algos.training import count_vars
from multiagent_rl.utils.logx import EpochLogger
from multiagent_rl.utils.evaluation_utils import *


def train_tournament(
    env_fn=RoundRobinTournament,
    env_kwargs=dict(),
    agent_fns=None,
    agent_kwargs=None,
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=10,
    logger_kwargs=dict(),
    save_freq=1,
):
    """
    Training loop for the multi-agent RoundRobinTournament environment. Note: many objects and functions are assumed
    to be implemented as agent attributes/methods whereas in single-agent RL they might be implemented in the overall
    training loop. (Such as buffers, optimizers, and update methods). In addition, many parameters (such as
    learning rates) are passed as keyword args to each agent.

    Args:
        env_fn: a function which creates a copy of the environment. Must satisfy the OpenAI Gym API.
        env_kwargs: keyword args for the environment constructor.
        agent_fns: a list of (two) constructor methods for the agents.
        agent_kwargs: a list of (two) dicts of keyword args for the agent constructors.
        seed: seed for random number generators.
        steps_per_epoch: number of interactions between the agents and environment in each epoch.
        epochs: total number of epochs to train agents over.
        batch_size: number of episodes per minibatch in optimization/SGD.
        start_steps: number of steps to perform (uniform) random actions before using agent policies.
            Intended for exploration.
        update_after: number of interaction updates to store to buffers before starting agent updates. Ensures there
            is enough data in buffers for updates to be useful.
        update_every: number of interactions to run between agent updates. Note: regardless of this value, the
            ratio of interactions to updates is set to 1.
        num_test_episodes: number of episodes to test deterministic agent policies at the end of each epoch.
        max_ep_len: max episode length. Note: This function assumes all episodes have a fixed length.
        logger_kwargs: keyword args for the logger.
        save_freq: how frequently (by number of epochs) to save current agents.
    """

    num_agents = len(agent_fns)
    env_kwargs['num_agents'] = num_agents
    assert len(agent_kwargs) == num_agents

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn(**env_kwargs)
    test_env = env_fn(**env_kwargs)

    env.seed(seed)
    test_env.seed(seed)
    for i, space in enumerate(env.action_space):
        space.seed(seed + i)
    for i, space in enumerate(test_env.action_space):
        space.seed(seed + i)

    agent_list = []
    for i, fn in enumerate(agent_fns):
        obs_space = env.observation_space[i]
        action_space = env.action_space[i]
        agent = fn(obs_space, action_space, **agent_kwargs[i])
        agent_list.append(agent)

    def deterministic_policy_test():
        for _ in range(num_test_episodes):
            obs = test_env.reset()
            for agent in agent_list:
                agent.reset_state()
            episode_return = np.zeros(num_agents)
            episode_length = 0
            done = False
            while not done and not episode_length == max_ep_len:
                act = [
                    agent_list[i].act(
                        torch.as_tensor(obs[i], dtype=torch.float32), noise=True
                    )
                    for i in range(num_agents)
                ]
                act = np.stack(act)
                obs, rwd, done, _ = test_env.step(act)
                episode_return += rwd
                episode_length += 1
                logger.store(TestActOffer1=act[0][0], TestActDemand1=act[0][1])
                logger.store(TestActOffer2=act[1][0], TestActDemand2=act[1][1])
            logger.store(TestEpRet1=episode_return[0], TestEpLen=episode_length)
            logger.store(TestEpRet2=episode_return[1])

    def reset_all():
        all_obs = env.reset()
        for agent in agent_list:
            agent.reset_state()
        episode_return = np.zeros(num_agents)
        episode_length = 0
        return all_obs, episode_return, episode_length

    start_time = time.time()
    # Begin training phase.
    t_total = 0
    for epoch in range(epochs):
        obs, episode_return, episode_length = reset_all()
        for t in range(steps_per_epoch):
            # Take random actions for first n steps to do broad exploration
            if t_total <= start_steps:
                act = [np.stack(a) for a in env.action_space.sample()]
            else:
                act = [
                    agent_list[i].act(
                        torch.as_tensor(obs[i], dtype=torch.float32), noise=True
                    )
                    for i in range(num_agents)
                ]
            act = np.stack(act)
            logger.store(ActOffer=act[0][0], ActDemand=act[0][1])
            # Step environment given latest agent action
            obs_next, rwd, done, _ = env.step(act)

            episode_return += rwd
            episode_length += 1

            # Store current step in buffers
            for i, agent in enumerate(agent_list):
                agent.store_to_buffer(obs[i], act[i], rwd[i], obs_next[i], done)

            # update episode return and env state
            obs = obs_next

            # check if episode is over
            if done:
                logger.store(EpRet=episode_return[0], EpLen=episode_length)
                obs, episode_return, episode_length = reset_all()

            if t_total >= update_after and t_total % update_every == 0:
                for i in range(update_every):
                    for agent in agent_list:
                        agent.update(batch_size, t_total)

            t_total += 1

        deterministic_policy_test()

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("ActOffer", with_min_and_max=True)
        logger.log_tabular("ActDemand", with_min_and_max=True)
        logger.log_tabular("TestEpRet1", with_min_and_max=True)
        logger.log_tabular("TestEpRet2", with_min_and_max=True)
        logger.log_tabular("TestActOffer1", with_min_and_max=True)
        logger.log_tabular("TestActOffer2", with_min_and_max=True)
        logger.log_tabular("TestActDemand1", with_min_and_max=True)
        logger.log_tabular("TestActDemand2", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("TestEpLen", average_only=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
