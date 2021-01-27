from itertools import chain
import time

from torch.optim import Adam

from multiagent_rl.algos.agents import *
from multiagent_rl.algos.buffers import *
from multiagent_rl.algos.training import count_vars
from multiagent_rl.utils.logx import EpochLogger
from multiagent_rl.utils.evaluation_utils import *


def two_agent_rsac(
    env_fn=None,
    env_kwargs=dict(),
    agent_fns=None,
    agent_kwargs=None,
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    max_buffer_len=100000,
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    a_lr=1e-3,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    log_interval=10,
    max_episode_len=10,
    logger_kwargs=dict(),
    save_freq=1,
    alpha=0.05,
    update_alpha_after=5000,
    target_entropy=-4.0,
    save_q_every=0,
    q_filename="/Users/kurtsmith/research/multiagent-rl/data/q",
):
    """Run Recurrent-SAC training."""
    # Initialize environment, agent, auxiliary objects

    num_agents = 2

    q1_filename = os.path.expanduser(q_filename) + "_1_map.csv"
    q2_filename = os.path.expanduser(q_filename) + "_2_map.csv"

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
        space.seed(seed+i)

    agent_list = []
    for i, fn in enumerate(agent_fns):
        obs_space = env.observation_space[i]
        action_space = env.action_space[i]
        agent =fn(obs_space, action_space, **agent_kwargs[i])
        agent_list.append(agent)

    def deterministic_policy_test():
        for _ in range(num_test_episodes):
            obs = test_env.reset()
            for agent in agent_list:
                agent.reset_state()
            episode_return = np.zeros(num_agents)
            episode_length = 0
            done = False
            while not done and not episode_length == max_episode_len:
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
                logger.store(TestActOffer=act[0][0], TestActDemand=act[0][1])
            logger.store(TestEpRet=episode_return[0], TestEpLen=episode_length)


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
    update_time = 0.0
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
        logger.log_tabular("TestEpRet", with_min_and_max=True)
        logger.log_tabular("TestActOffer", with_min_and_max=True)
        logger.log_tabular("TestActDemand", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("TestEpLen", average_only=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        # logger.log_tabular("Q1Vals", with_min_and_max=True)
        # logger.log_tabular("Q2Vals", with_min_and_max=True)
        # logger.log_tabular("LossPi", average_only=True)
        # logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
