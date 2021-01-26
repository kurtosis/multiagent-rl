from itertools import chain
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
    env.action_space.seed(seed)
    test_env.seed(seed)
    test_env.action_space.seed(seed)

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
    # q_params = chain(agent.q1.parameters(), agent.q2.parameters())

    # Freeze target mu, Q so they are not updated by optimizers
    # for p in agent_target.parameters():
    #     p.requires_grad = False

    var_counts = tuple(count_vars(module) for module in [agent.pi, agent.q1])
    logger.log(
        f"\nNumber of parameters \t policy: {var_counts[0]} q1/2: {var_counts[1]}\n"
    )

    # Create agents, either all same type or specified individually
    agent_action_space = env.action_space[0][0]
    agent_act_dim = agent_action_space.shape[0]
    num_agents = len(agent_fns)
    if agent_kwargs is None:
        agent_list = [
            fn(obs_dim=agent_obs_dim, action_space=agent_action_space)
            for fn in agent_fns
        ]
    else:
        agent_list = [
            agent_fn(obs_dim=agent_obs_dim, action_space=agent_action_space, **kwargs)
            for agent_fn, kwargs in zip(agents, agents_kwargs)
        ]

    multi_buf = MultiagentRDPGBuffer(
        agent_obs_dim, agent_act_dim, num_agents, replay_size
    )

    def deterministic_policy_test():
        for _ in range(num_test_episodes):
            obs = test_env.reset()
            agent.reset_state()
            ep_ret = 0
            ep_len = 0
            done = False
            while not done and not ep_len == max_episode_len:
                act = agent.act(
                    torch.as_tensor(obs, dtype=torch.float32), deterministic=True
                )
                obs, rwd, done, _ = test_env.step(act)
                ep_ret += rwd
                ep_len += 1
                logger.store(TestActOffer=act[0], TestActDemand=act[1])
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()

    def reset_all():
        all_obs = env.reset()
        for agent in agent_list:
            agent.reset_state()
        episode_return = np.zeros(num_agents)
        episode_length = 0
        return all_obs, episode_return, episode_length

    # Begin training phase.
    t_total = 0
    update_time = 0.0
    for epoch in range(epochs):
        all_obs, episode_return, episode_length = reset_all()
        for t in range(steps_per_epoch):
            # Take random actions for first n steps to do broad exploration
            if t_total <= start_steps:
                actions = [np.stack(a) for a in env.action_space.sample()]
                actions = np.concatenate(actions)
            else:
                actions = [
                    agent_list[i].act(
                        torch.as_tensor(all_obs[i], dtype=torch.float32), noise=True
                    )
                    for i in range(num_agents)
                ]
                actions = np.stack(actions)

            # Step environment given latest agent action
            all_obs_next, rwd, done, _ = env.step(actions)

            episode_return += rwd
            episode_length += 1

            # Store current step in buffer
            buf.store(all_obs, actions, rwd, all_obs_next, done)

            # update episode return and env state
            all_obs = obs_next

            # check if episode is over
            if done:
                logger.store(EpRet=episode_return, EpLen=episode_length)
                all_obs, episode_return, episode_length = reset_all()

            if t_total >= update_after and t_total % update_every == 0:
                update_start = time.time()
                for i in range(update_every):
                    for agent in agent_list:
                        agent.update()
                    # update()
                    # alpha = log_alpha.exp()

            t_total += 1

        deterministic_policy_test()

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        logger.log_tabular("ActOffer", with_min_and_max=True)
        logger.log_tabular("ActDemand", with_min_and_max=True)
        logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("TestActOffer", with_min_and_max=True)
        logger.log_tabular("TestActDemand", with_min_and_max=True)
        logger.log_tabular("TestEpRet", with_min_and_max=True)
        logger.log_tabular("EpLen", average_only=True)
        logger.log_tabular("TestEpLen", average_only=True)
        logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("Q1Vals", with_min_and_max=True)
        logger.log_tabular("Q2Vals", with_min_and_max=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
