from multiagent_rl.algos.agents import *
from multiagent_rl.algos.training import count_vars
from multiagent_rl.utils.logx import EpochLogger
from multiagent_rl.algos.buffers import *
from multiagent_rl.algos.orig_ddpg.ddpg import ReplayBuffer
from multiagent_rl.utils.evaluation_utils import *


def ddpg_new(
    env_fn,
    agent_fn=DDPGAgent,
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    replay_size=1000000,
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    save_q_every=1000,
    num_test_episodes=10,
    log_interval=10,
    max_episode_len=1000,
    agent_kwargs=dict(),
    env_kwargs=dict(),
    logger_kwargs=dict(),
    save_freq=1,
    q_filename="~/q_map_ddpg_noise_5.csv",
):
    """Run DDPG training."""
    # Initialize environment, agent, auxiliary objects

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
    act_dim = env.action_space.shape
    agent = agent_fn(
        observation_space=env.observation_space,
        action_space=env.action_space,
        **agent_kwargs,
    )
    agent_target = deepcopy(agent)

    # Freeze target mu, Q so they are not updated by optimizers
    for p in agent_target.parameters():
        p.requires_grad = False

    var_counts = tuple(count_vars(module) for module in [agent.pi, agent.q])
    logger.log(
        f"\nNumber of parameters \t policy: {var_counts[0]} q: {var_counts[1]}\n"
    )

    # buf = TransitionBuffer(obs_dim, act_dim, replay_size)
    buf = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    pi_optimizer = Adam(agent.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(agent.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(agent)

    def compute_loss_pi(data):
        o = data["obs"]
        q_pi = agent.q(torch.cat([o, agent.pi(o)], dim=-1))
        return -q_pi.mean()

    # Set up function for computing DDPG Q-loss
    def compute_loss_q(data):
        o, a, r, o2, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )

        q = agent.q(torch.cat([o, a], dim=-1))

        # Bellman backup for Q function
        with torch.no_grad():
            # q_pi_targ = ac_targ.q(o2, ac_targ.pi(o2))
            a2 = agent_target.pi(o2)
            q_pi_targ = agent_target.q(torch.cat([o2, a2], dim=-1))
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q = ((q - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(QVals=q.detach().numpy())

        return loss_q, loss_info

    # q_time = 0.0
    # pi_time = 0.0
    # target_time = 0.0
    # batch_time = 0.0

    # def update(q_time, pi_time, target_time):
    def update():
        # Get training data from buffer
        # data = buf.get(batch_size=batch_size)
        data = buf.sample_batch(batch_size)

        # Update Q function
        # t0 = time.time()
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        # t1 = time.time()
        # q_time += (t1-t0)

        # Freeze Q params during policy update to save time
        for p in agent.q.parameters():
            p.requires_grad = False
        # Update policy
        pi_optimizer.zero_grad()
        loss_pi = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q params after policy update
        for p in agent.q.parameters():
            p.requires_grad = True
        # t2 = time.time()
        # pi_time += (t2-t1)

        logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

        with torch.no_grad():
            for p, p_target in zip(agent.parameters(), agent_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)
        # t3 = time.time()
        # target_time += (t3-t2)

        # return q_time, pi_time, target_time

    def deterministic_policy_test():
        for _ in range(num_test_episodes):
            o = test_env.reset()
            ep_ret = 0
            ep_len = 0
            d = False
            while not d and not ep_len == max_episode_len:
                # with torch.no_grad():
                a = agent.act(torch.as_tensor(o, dtype=torch.float32), noise=False)
                logger.store(TestActOffer=a[0], TestActDemand=a[1])
                o, r, d, _ = test_env.step(a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    start_time = time.time()

    # Begin training phase.
    t_total = 0
    update_time = 0.0
    for epoch in range(epochs):
        obs = env.reset()
        episode_return = 0
        episode_length = 0
        for t in range(steps_per_epoch):
            # Take random actions for first n steps to do broad exploration
            if t_total <= start_steps:
                act = env.action_space.sample()
            else:
                act = agent.act(torch.as_tensor(obs, dtype=torch.float32), noise=True)
            logger.store(ActOffer=act[0], ActDemand=act[1])
            # Step environment given latest agent action
            obs_next, reward, done, _ = env.step(act)

            episode_return += reward
            episode_length += 1

            # Store current step in buffer
            buf.store(obs, act, reward, obs_next, done)

            # update episode return and env state
            obs = obs_next

            # check if episode is over
            episode_capped = episode_length == max_episode_len
            epoch_ended = t == steps_per_epoch - 1
            end_episode = done or episode_capped or epoch_ended
            if end_episode:
                # episode_count += 1
                if done or episode_capped:
                    logger.store(EpRet=episode_return, EpLen=episode_length)
                obs = env.reset()
                episode_return = 0
                episode_length = 0

            if t_total >= update_after and t % update_every == 0:
                update_start = time.time()
                for _ in range(update_every):
                    update()
                    # q_time, pi_time, target_time = update(q_time, pi_time, target_time)
                    # n_updates += 1
                update_end = time.time()
                update_time += update_end - update_start
                total_time = update_end - start_time
                # print(f't total {t_total}; t {t}; epoch {epoch}')
                # print(f'update time {update_time}')
                # print(f'total time {total_time}')
                # print('---')

            if t_total >= update_after and t_total % save_q_every == 0:
                save_q_map(agent.q, q_filename, step=t_total)
            t_total += 1

        deterministic_policy_test()
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

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
        logger.log_tabular("QVals", with_min_and_max=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
        # print(f'update_time {update_time}')
        # print(f'q_time {q_time}')
        # print(f'pi_time {pi_time}')
        # print(f'target_time {target_time}')

        # Look at pi, q functions
        # batch = buf.sample_batch(batch_size)
        # eval_q_vs_a_2(batch, agent)
        # eval_a(batch, agent)
