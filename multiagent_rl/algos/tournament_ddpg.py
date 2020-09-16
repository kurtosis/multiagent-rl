import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import sys

sys.path.insert(
    0, "/Users/kurtsmith/research/pytorch_projects/reinforcement_learning/environments"
)
sys.path.insert(0, "/Users/kurtsmith/research/spinningup")

import pandas as pd

from gym.spaces import Box, Discrete, Tuple

from multiagent_rl.algos.agents import *
from multiagent_rl.algos.buffers import *
from multiagent_rl.environments.tournament_env import *
from multiagent_rl.utils.logx import EpochLogger, EpisodeLogger


def eval_q_fn(agent, nn=101, filename="/Users/kurtsmith/q.csv"):
    xx = np.linspace(0, 1, nn)
    yy = np.linspace(0, 1, nn)
    xg, yg = np.meshgrid(xx, yy)
    xf = np.reshape(xg, (nn * nn, 1))
    yf = np.reshape(yg, (nn * nn, 1))
    grid_pts = np.concatenate((xf, yf), axis=-1)
    obs = np.zeros((grid_pts.shape[0], 11))
    grid_input = torch.tensor(
        np.concatenate((obs, grid_pts), axis=-1), dtype=torch.float32
    )
    q_long = np.round(agent.q(grid_input).detach().numpy(), 4)
    # q_grid = np.reshape(q_long, xg.shape)
    # np.savetxt(filename, q_grid, fmt='%2.4f', delimiter=', ')
    df_q = pd.DataFrame(
        dict(offer=xf.squeeze(), threshold=yf.squeeze(), q=q_long.squeeze())
    )
    df_q.to_csv(filename, index=False)


def tournament_ddpg(
    agent_fn=DDPGAgent,
    agent_kwargs=dict(),
    num_agents=4,
    agents=None,
    agents_kwargs=None,
    env_fn=RoundRobinTournament,
    env_kwargs=dict(),
    seed=0,
    epochs=10,
    steps_per_epoch=5000,
    replay_size=1000000,
    sample_size=100,
    start_steps=10,
    update_after=1000,
    update_every=50,
    test_episodes=100,
    log_interval=10,
    max_episode_len=1000,
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    logger_kwargs=dict(),
    q_file=None,
    save_freq=10,
):
    """Run DDPG training."""
    # Initialize environment, agent, auxiliary objects

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    episode_logger = EpisodeLogger(num_agents=num_agents, **logger_kwargs)
    test_logger = EpochLogger(**logger_kwargs)

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn(num_agents=num_agents, **env_kwargs)
    test_env = env_fn(num_agents=num_agents, **env_kwargs)

    # Determine obs_dim, # of observation vars
    if isinstance(env.observation_space[0], Box):
        agent_obs_dim = env.observation_space[0].shape[0]
    elif isinstance(env.observation_space[0], Discrete):
        agent_obs_dim = 1
    elif isinstance(env.observation_space[0], Tuple):
        agent_obs_dim = 0
        for space in env.observation_space[0]:
            if isinstance(space, Box):
                agent_obs_dim += space.shape[0]
            else:
                agent_obs_dim += 1

    # Create agents, either all same type or specified individually
    agent_action_space = env.action_space[0][0]
    agent_act_dim = agent_action_space.shape[0]
    if agents is None:
        agent_list = [
            agent_fn(
                obs_dim=agent_obs_dim, action_space=agent_action_space, **agent_kwargs
            )
        ] * num_agents
    else:
        num_agents = len(agents)
        if agents_kwargs is None:
            agent_list = [
                agent_fn(obs_dim=agent_obs_dim, action_space=agent_action_space)
                for agent_fn in agents
            ]
        else:
            agent_list = [
                agent_fn(
                    obs_dim=agent_obs_dim, action_space=agent_action_space, **kwargs
                )
                for agent_fn, kwargs in zip(agents, agents_kwargs)
            ]

    multi_buf = MultiagentTransitionBuffer(
        agent_obs_dim, agent_act_dim, num_agents, replay_size
    )

    # Set up model saving
    # logger.setup_pytorch_saver(agent_1)

    def deterministic_policy_test():
        for i_epi in range(test_episodes):
            all_obs = test_env.reset()
            episode_return = np.zeros(num_agents)
            episode_length = 0
            done = False
            log_episode = (epoch % save_freq == 0) or (epoch == epochs - 1)
            while not done and not episode_length == max_episode_len:
                with torch.no_grad():
                    actions = [
                        agent_list[i].act(
                            torch.as_tensor(all_obs[i], dtype=torch.float32),
                            noise=False,
                        )
                        for i in range(num_agents)
                    ]
                    actions = np.stack(actions)

                logger.store(
                    TestOffer0=actions[0, 0],
                    TestThreshold0=actions[0, 1],
                    TestOffer1=actions[1, 0],
                    TestThreshold1=actions[1, 1],
                )
                if test_env.current_round == test_env.num_rounds:
                    logger.store(
                        TestOfferFirstRound=actions[0, 0],
                        TestThresholdFirstRound=actions[0, 1],
                    )
                if test_env.current_round == 1:
                    logger.store(
                        TestOfferLastRound=actions[0, 0],
                        TestThresholdLastRound=actions[0, 1],
                    )

                if log_episode:
                    episode_logger.store(epoch=epoch,
                                         episode=i_epi,
                                         action=actions.flatten(),
                                         **test_env.get_state()
                                         )

                all_obs, reward, done, _ = test_env.step(actions)

                if log_episode:
                    episode_logger.store(
                                     reward=reward,
                                     done=done,
                                     )

                episode_return += reward
                episode_length += 1

            if log_episode:
                episode_logger.dump_dataframe()

            logger.store(
                TestEpRet0=episode_return[0],
                TestEpRet1=episode_return[1],
                TestEpRet2=episode_return[2],
                TestEpRet3=episode_return[3],
                TestEpLen=episode_length,
                TestEpScore0=test_env.scores[0],
                TestEpScore1=test_env.scores[1],
                TestEpScore2=test_env.scores[2],
                TestEpScore3=test_env.scores[3],
                TestMeanScore=np.mean(test_env.scores),
                TestStdScore=np.std(test_env.scores),
                TestMaxScore=np.max(test_env.scores),
                TestMinScore=np.min(test_env.scores),
            )

    start_time = time.time()

    # Begin training phase.
    t_total = 0
    # play_time = 0.0
    # update_time = 0.0
    # agent_time = 0.0
    # deterministic_time = 0.0
    for epoch in range(epochs):
        all_obs = env.reset()
        episode_return = np.zeros(num_agents)
        episode_length = 0
        episode_count = 0
        for t in range(steps_per_epoch):
            # play_start = time.time()
            if t_total < start_steps:
                # Randomly sample actions. Note: this "cheats" and treats agents as interchangeable (same act space)
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

            logger.store(
                Offer0=actions[0, 0], Threshold0=actions[0, 1],
            )
            # for p in agent_list[0].pi.parameters():
            #     print(p.data)
            #     print('--')
            # print('------')
            all_obs_next, reward, done, _ = env.step(actions)
            episode_return += reward
            episode_length += 1
            # Store current step in buffer
            multi_buf.store(all_obs, actions, reward, all_obs_next, done)

            # update episode return and env state
            all_obs = all_obs_next
            # play_end = time.time()
            # play_time += (play_end - play_start)

            # check if episode is over
            episode_capped = episode_length == max_episode_len
            epoch_ended = t == steps_per_epoch - 1
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                episode_count += 1
                if done or episode_capped:
                    # print(f"end {env.scores}")
                    # print(f"round {env.current_round}")
                    # print(f"turn {env.current_turn}")
                    logger.store(
                        EpRet0=episode_return[0],
                        EpRet1=episode_return[1],
                        EpRet2=episode_return[2],
                        EpRet3=episode_return[3],
                        EpLen=episode_length,
                        EpScore0=env.scores[0],
                        EpScore1=env.scores[1],
                        EpScore2=env.scores[2],
                        EpScore3=env.scores[3],
                    )
                all_obs = env.reset()
                # print(f"start {env.scores}")
                episode_return = np.zeros(num_agents)
                episode_length = 0

                if t_total >= update_after and (t + 1) % update_every == 0:
                    # update_start = time.time()
                    for _ in range(update_every):
                        data = multi_buf.get(sample_size=sample_size)

                        def slicer(v, i):
                            if v.dim() == 1:
                                return v
                            elif v.dim() == 2:
                                return v[:, i]
                            else:
                                return v[:, i, :]

                        for i in range(num_agents):
                            data_agent = {k: slicer(v, i) for k, v in data.items()}
                            agent_list[i].update(data_agent, logger=logger)

            t_total += 1
        logger.store(NumEps=episode_count)

        # det_start = time.time()
        deterministic_policy_test()
        # det_end = time.time()
        # deterministic_time += (det_end - det_start)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)
            if q_file is not None:
                eval_q_fn(
                    agent_list[0], filename=f"{logger.output_dir}/{q_file}_{epoch}.csv"
                )

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        # logger.log_tabular("EpRet0", average_only=True)
        logger.log_tabular("EpRet0", with_min_and_max=True)
        # logger.log_tabular("EpRet1", average_only=True)
        # logger.log_tabular("EpRet2", average_only=True)
        # logger.log_tabular("EpRet3", average_only=True)
        logger.log_tabular("EpScore0", average_only=True)
        logger.log_tabular("Offer0", with_min_and_max=True)
        logger.log_tabular("Threshold0", with_min_and_max=True)
        # logger.log_tabular("EpScore1", average_only=True)
        # logger.log_tabular("EpScore2", average_only=True)
        # logger.log_tabular("EpScore3", average_only=True)
        logger.log_tabular("TestEpRet0", average_only=True)
        # logger.log_tabular("TestEpRet1", average_only=True)
        # logger.log_tabular("TestEpRet2", average_only=True)
        # logger.log_tabular("TestEpRet3", average_only=True)
        logger.log_tabular("TestEpScore0", with_min_and_max=True)
        # logger.log_tabular("TestEpScore1", with_min_and_max=True)
        # logger.log_tabular("TestEpScore2", with_min_and_max=True)
        # logger.log_tabular("TestEpScore3", with_min_and_max=True)
        logger.log_tabular("TestOffer0", with_min_and_max=True)
        logger.log_tabular("TestThreshold0", with_min_and_max=True)
        # logger.log_tabular("Offer1", with_min_and_max=True)
        # logger.log_tabular("Threshold1", with_min_and_max=True)
        # logger.log_tabular("TestMeanScore", average_only=True)
        # logger.log_tabular("TestStdScore", average_only=True)
        # logger.log_tabular("TestMaxScore", average_only=True)
        # logger.log_tabular("TestMinScore", average_only=True)
        # logger.log_tabular("TestOfferFirstRound", with_min_and_max=True)
        # logger.log_tabular("TestThresholdFirstRound", with_min_and_max=True)
        # logger.log_tabular("TestOfferLastRound", with_min_and_max=True)
        # logger.log_tabular("TestThresholdLastRound", with_min_and_max=True)
        # logger.log_tabular("NumEps", average_only=True)
        # logger.log_tabular("EpLen", average_only=True)
        # logger.log_tabular("TestEpLen", average_only=True)
        # logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        logger.log_tabular("QVals", with_min_and_max=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
        # print(f'play_time {play_time}')
        # print(f'update_time {update_time}')
        # print(f'agent_time {agent_time}')
        # print(f'deterministic_time {deterministic_time}')
