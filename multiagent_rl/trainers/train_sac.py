from itertools import chain
import time

from torch.optim import Adam

from multiagent_rl.agents.agents import *
from multiagent_rl.buffers import TransitionBuffer
from multiagent_rl.trainers.train_rsac import count_vars
from multiagent_rl.utils.logx import EpochLogger
from multiagent_rl.utils.evaluation_utils import *


def train_sac(
    env_fn,
    env_kwargs=dict(),
    agent_fn=SACAgent,
    agent_kwargs=dict(),
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    max_episode_len=1000,
    max_buffer_size=1000000,
    pi_lr=1e-3,
    q_lr=1e-3,
    a_lr=1e-3,
    gamma=0.99,
    polyak=0.995,
    alpha=0.05,
    update_alpha_after=5000,
    target_entropy=-4.0,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    logger_kwargs=dict(),
    save_freq=1,
):
    """
    Training loop for a single-agent environment, using transition-based training suitable for SACAgent,
    but not RSACAgent.

    Args:
        env_fn: a function which creates a copy of the environment. Must satisfy the OpenAI Gym API.
        env_kwargs: keyword args for the environment constructor.
        agent_fn: constructor methods for the agent.
        agent_kwargs: dict of keyword args for the agent constructor.
        seed: seed for random number generators.
        steps_per_epoch: number of interactions between the agent and environment in each epoch.
        epochs: total number of epochs to train agents over.
        max_episode_len: max episode length, used in buffer size
        max_buffer_size: max number of transitions to store in buffer
        pi_lr: learning rate for pi updates
        q_lr: learning rate for Q updates
        a_lr: learning rate for alpha updates
        gamma: discount factor (between 0 and 1) for Q updates
        polyak: interpolation factor for Q target updates (between 0 and 1, typically close to 1)
        alpha: entropy regularization coefficient (larger values penalize low-entropy pi)
        update_alpha_after : number of env interactions to run before updating alpha
        target_entropy : controls the min value that alpha is reduced to during training.
            (typically negative, lower values cause alpha to be reduced more)
        batch_size: number of episodes per minibatch in optimization/SGD.
        start_steps: number of steps to perform (uniform) random actions before using agent policies.
            Intended for exploration.
        update_after: number of interaction updates to store to buffers before starting agent updates. Ensures there
            is enough data in buffers for updates to be useful.
        update_every: number of interactions to run between agent updates. Note: regardless of this value, the
            ratio of interactions to updates is set to 1.
        num_test_episodes: number of episodes to test deterministic agent policies at the end of each epoch.
        logger_kwargs: keyword args for the logger.
        save_freq: how frequently (by number of epochs) to save current agent.
    """

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
    agent = agent_fn(
        obs_space=env.observation_space, action_space=env.action_space, **agent_kwargs,
    )
    agent_target = deepcopy(agent)

    log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
    q_params = chain(agent.q1.parameters(), agent.q2.parameters())

    # Freeze target mu, Q so they are not updated by optimizers
    for p in agent_target.parameters():
        p.requires_grad = False

    var_counts = tuple(count_vars(module) for module in [agent.pi, agent.q1])
    logger.log(
        f"\nNumber of parameters \t policy: {var_counts[0]} q1/2: {var_counts[1]}\n"
    )

    buf = TransitionBuffer(obs_dim, act_dim, max_buffer_size)

    pi_optimizer = Adam(agent.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)
    alpha_optimizer = Adam([log_alpha], lr=a_lr)

    # Set up model saving
    logger.setup_pytorch_saver(agent)

    def compute_loss_pi(data):
        obs = data["obs"]
        act, logprob_pi = agent.pi(obs)
        pi_info = dict(LogPi=logprob_pi.detach().numpy())

        q1_pi = agent.q1(torch.cat([obs, act], dim=-1))
        q2_pi = agent.q2(torch.cat([obs, act], dim=-1))
        q_pi = torch.min(q1_pi, q2_pi)

        loss_pi = (-q_pi + alpha * logprob_pi).mean()

        return loss_pi, pi_info

    def compute_loss_q(data):
        obs, act, rwd, obs_next, done = (
            data["obs"],
            data["act"],
            data["rwd"],
            data["obs_next"],
            data["done"],
        )

        q1 = agent.q1(torch.cat([obs, act], dim=-1))
        q2 = agent.q2(torch.cat([obs, act], dim=-1))

        # Bellman backup for Q function
        with torch.no_grad():
            a_next, logprob_next = agent.pi(obs_next)

            q1_target = agent_target.q1(torch.cat([obs_next, a_next], dim=-1))
            q2_target = agent_target.q2(torch.cat([obs_next, a_next], dim=-1))
            q_target = torch.min(q1_target, q2_target)
            backup = rwd + gamma * (1 - done) * (q_target - alpha * logprob_next)

        # MSE loss against Bellman backup
        loss_q = ((q1 - backup) ** 2).mean() + ((q2 - backup) ** 2).mean()

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(), Q2Vals=q2.detach().numpy(),)

        return loss_q, loss_info

    def update():
        # Get training data from buffer
        data = buf.sample_batch(batch_size)

        # Update Q function
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        logger.store(LossQ=loss_q.item(), **loss_info)

        for p in q_params:
            p.requires_grad = False

        # Update policy
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q params after policy update
        for p in q_params:
            p.requires_grad = True

        logger.store(LossPi=loss_pi.item(), **pi_info)

        with torch.no_grad():
            for p, p_target in zip(agent.parameters(), agent_target.parameters()):
                p_target.data.mul_(polyak)
                p_target.data.add_((1 - polyak) * p.data)

        # Update alpha
        if t_total >= update_alpha_after:
            obs = data["obs"]
            pi, log_pi = agent.pi(obs)
            loss_alpha = (log_alpha * (-log_pi - target_entropy).detach()).mean()
            alpha_optimizer.zero_grad()
            loss_alpha.backward()
            alpha_optimizer.step()

    def deterministic_policy_test():
        for _ in range(num_test_episodes):
            obs = test_env.reset()
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
                act = agent.act(torch.as_tensor(obs, dtype=torch.float32))
            logger.store(ActOffer=act[0], ActDemand=act[1])
            # Step environment given latest agent action
            obs_next, rwd, done, _ = env.step(act)

            episode_return += rwd
            episode_length += 1

            # Store current step in buffer
            buf.store(obs, act, rwd, obs_next, done)

            # update episode return and env state
            obs = obs_next

            # check if episode is over
            if done:
                logger.store(EpRet=episode_return, EpLen=episode_length)

            if t_total >= update_after and t_total % update_every == 0:
                update_start = time.time()
                for i in range(update_every):
                    update()
                    alpha = log_alpha.exp()

            t_total += 1

        deterministic_policy_test()

        # Save model at end of epoch
        if t == steps_per_epoch - 1:
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
        logger.log_tabular("Q1Vals", with_min_and_max=True)
        logger.log_tabular("Q2Vals", with_min_and_max=True)
        logger.log_tabular("LossPi", average_only=True)
        logger.log_tabular("LossQ", average_only=True)
        logger.log_tabular("Time", time.time() - start_time)
        logger.dump_tabular()
