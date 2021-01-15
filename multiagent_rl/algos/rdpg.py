from multiagent_rl.algos.agents import *
from multiagent_rl.algos.training import count_vars
from multiagent_rl.utils.logx import EpochLogger
from multiagent_rl.algos.buffers import *

def rdpg(
    env_fn,
    agent_fn=RDPGAgent,
    seed=0,
    epochs=100,
    steps_per_epoch=4000,
    max_buffer_len=100000,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    test_episodes=10,
    log_interval=10,
    max_episode_len=3,
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    agent_kwargs=dict(),
    env_kwargs=dict(),
    logger_kwargs=dict(),
    save_freq=10,
):
    """Run RDPG training."""
    # Initialize environment, agent, auxiliary objects

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)


    data_time = 0.0
    q_time = 0.0
    pi_time = 0.0
    target_time = 0.0
    batch_time = 0.0

    env = env_fn(**env_kwargs)
    test_env = deepcopy(env)
    # test_env = env_fn(**env_kwargs)
    # env.seed(seed)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    agent = agent_fn(
        observation_space=env.observation_space, action_space=env.action_space, **agent_kwargs
    )
    agent_target = deepcopy(agent)

    # Freeze target mu, Q so they are not updated by optimizers
    for p in agent_target.parameters():
        p.requires_grad = False

    if hasattr(agent, 'q'):
        var_counts = tuple(count_vars(module) for module in [agent.pi, agent.q])
    else:
        var_counts = tuple(count_vars(module) for module in [agent.pi, agent.q1])
    logger.log(
        f"\nNumber of parameters \t policy: {var_counts[0]} q: {var_counts[1]}\n"
    )

    # buf = RDPGBuffer(max_buffer_len)
    buf = RDPGBuffer(obs_dim, act_dim, max_episode_len, max_buffer_len)
    pi_optimizer = Adam(agent.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(agent.q.parameters(), lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(agent)

    def compute_loss_policy(data):
        # get data
        o = data["obs"]
        # Get actions that agent policy would take at each step
        a = agent.pi(o)
        return -agent.q(torch.cat((o, a), dim=-1)).mean()

    def compute_q_target(data):
        r, o_next, d = data["rwd"], data["obs_next"], data["done"]
        o = data['obs']
        with torch.no_grad():
            o_padded = pad(o, (0, 0, 0, 0, 0, 1), "constant", 0)
            a_padded = agent_target.pi(o_padded)
            q_target_padded = agent_target.q(torch.cat((o_padded, a_padded), dim=-1))

            # HACK - probably a way to do this in one line with slice?
            if len(q_target_padded.shape)==3:
                q_target = q_target_padded[1:, :, :]
            elif len(q_target_padded.shape)==2:
                q_target = q_target_padded[1:, :]
            else:
                raise ValueError('Q tensor has unexpected number of dimensions!')

            # reshape so that this will work with DDPG agent (for testing)
            q_target = q_target.reshape_as(d)
            q_target = r + gamma * (1 - d) * q_target
            # TESTING HACK - set all future q to 0 (assumes known optimal policy)
            # q_target = r
        return q_target

    def compute_loss_q(data, q_target):
        o, a = data["obs"], data["act"]
        q = agent.q(torch.cat((o, a), dim=-1))
        q_loss_info = {"QVals": q.detach().numpy()}
        q = q.reshape_as(q_target)
        return ((q - q_target) ** 2).mean(), q_loss_info

    # def update(data_time, q_time, pi_time, target_time):
    def update():
        # Get training data from buffer
        t0_data = time.time()
        data = buf.sample_episodes(batch_size=batch_size)
        t1_data = time.time()
        # data_time += (t1_data-t0_data)

    # Update Q function
        t0 = time.time()
        q_optimizer.zero_grad()
        q_target = compute_q_target(data)
        q_loss, q_loss_info = compute_loss_q(data, q_target)
        q_loss.backward()
        q_optimizer.step()
        t1 = time.time()
        # q_time += (t1-t0)

        # Freeze Q params during policy update to save time
        for p in agent.q.parameters():
            p.requires_grad = False
        # Update policy
        pi_optimizer.zero_grad()
        pi_loss = compute_loss_policy(data)
        pi_loss.backward()
        pi_optimizer.step()
        # Unfreeze Q params after policy update
        for p in agent.q.parameters():
            p.requires_grad = True
        t2 = time.time()
        # pi_time += (t2-t1)

        with torch.no_grad():
            for p, p_target in zip(agent.parameters(), agent_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)
        t3 = time.time()
        # target_time += (t3-t2)

        logger.store(LossPi=pi_loss.item(), LossQ=q_loss.item(), **q_loss_info)
        return data_time, q_time, pi_time, target_time

    def deterministic_policy_test():
        for _ in range(test_episodes):
            o = test_env.reset()
            agent.reset_state()
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
    update_time = 0.0
    for epoch in range(epochs):
        obs = env.reset()
        agent.reset_state()
        buf.clear_current_episode()
        episode_return = 0
        episode_length = 0
        pass
        for t in range(steps_per_epoch):
            # Take random actions for first n steps to do broad exploration
            if t_total < start_steps:
                act = env.action_space.sample()
            else:
                act = agent.act(torch.as_tensor(obs, dtype=torch.float32), noise=True)
            # Step environment given latest agent action
            obs_next, reward, done, _ = env.step(act)

            episode_return += reward
            episode_length += 1

            # Store current step in buffer
            buf.store(obs, act, reward, obs_next, np.array([done]))

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
                agent.reset_state()
                episode_return = 0
                episode_length = 0

            if t_total >= update_after and (t + 1) % update_every == 0:
                update_start = time.time()
                if epoch % 3 == 0:
                    pass
                for _ in range(update_every):
                    update()
                    # data_time, q_time, pi_time, target_time = update(data_time, q_time, pi_time, target_time)
                    # n_updates += 1
                # update_end = time.time()
                # update_time += (update_end - update_start)
                # total_time = update_end - start_time
                # print(f't_total {t_total}; t {t}; epoch {epoch}')
                # print(f'update_time {update_time}')
                # print(f'data_time {data_time}')
                # print(f'q_time {q_time}')
                # print(f'pi_time {pi_time}')
                # print(f'target_time {target_time}')
                # print(f'total time {total_time}')
                # print('---')

            t_total += 1

        deterministic_policy_test()
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({"env": env}, None)

        # Log info about epoch
        logger.log_tabular("Epoch", epoch)
        # logger.log_tabular("EpRet", with_min_and_max=True)
        logger.log_tabular("TestEpRet", with_min_and_max=True)
        # logger.log_tabular("EpLen", average_only=True)
        # logger.log_tabular("TestEpLen", average_only=True)
        # logger.log_tabular("TotalEnvInteracts", (epoch + 1) * steps_per_epoch)
        # logger.log_tabular("QVals", with_min_and_max=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
        # print(f'update_time {update_time}')
        # print(f'q_time {q_time}')
        # print(f'pi_time {pi_time}')
        # print(f'target_time {target_time}')


