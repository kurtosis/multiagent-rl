from torch.nn.functional import pad

from multiagent_rl.algos.rdpg import *
from multiagent_rl.environments.tournament_env import *


def test_rdpg_buffer(
    obs_dim=5,
    act_dim=2,
    max_buffer_len=100000,
    ep_len=3,
    num_ep=1000,
    sample_size=100,
    verbose=False,
):
    """Fill an RDPG buffer with sample data and verify it can be read from correctly."""
    buffer = RDPGBuffer(max_buffer_len)
    t0 = time.time()
    for i_ep in range(num_ep):
        for i_turn in range(ep_len):
            # create random observations and flag for first turn
            if i_turn == 0:
                obs = np.zeros(obs_dim - 1)
                obs = np.append(obs, 1)
            else:
                obs = np.random.rand(obs_dim - 1)
                obs = np.append(obs, 0)
            # Set done flag to 1 for last turn
            if i_turn == ep_len - 1:
                done = 1
            else:
                done = 0
            act = np.random.rand(act_dim)
            obs_next = np.random.rand(obs_dim - 1)
            obs_next = np.append(obs, 0)
            rwd = np.random.rand(1)
            buffer.store(obs, act, rwd, obs_next, done)
    t1 = time.time()
    time_fill = t1 - t0
    if verbose:
        print("Buffer filling completed")
        print(f"# of episodes in buffer: {buffer.filled_size}")
        print(f"Is buffer full: {buffer.full}")
        print(f"First episode: {buffer.episodes[0]}")
        if not buffer.full:
            print(f"Empty episode: {buffer.episodes[num_ep]}")
    t0 = time.time()
    sampled_episodes = buffer.sample_episodes(sample_size=sample_size)
    t1 = time.time()
    time_sample = t1 - t0
    if verbose:
        print(f"Time to fill (sec): {time_fill}")
        print(f"Time to sample (sec): {time_sample}")


def train_LSTMEstimator(
    input_size=2,
    hidden_size=8,
    max_buffer_len=100000,
    q_lr=1e-3,
    num_ep=10000,
    ep_len=3,
    update_every=10,
    gamma=0.99,
    polyak=0.995,
    use_target_fn=False,
    goal_constant=None,
    goal_mean=False,
):
    q = LSTMEstimator(input_size, hidden_size)
    q_target_copy = deepcopy(q)
    # Freeze target Q so they are not updated by optimizers
    for p in q_target_copy.parameters():
        p.requires_grad = False
    buf = RDPGBuffer(max_buffer_len)
    q_optimizer = Adam(q.parameters(), lr=q_lr)
    env = MimicObs(ep_len=ep_len, goal_constant=goal_constant, goal_mean=goal_mean)

    if goal_constant is not None:

        def best_action(obs, obs_next):
            return goal_constant * torch.ones_like(obs_next)

    elif not goal_mean:

        def best_action(obs, obs_next):
            return obs_next

    else:

        p3d = (0, 0, 0, 0, 0, 1)

        def best_action(obs, obs_next):
            cums = torch.cumsum(obs, 0) / torch.range(1, ep_len).view(ep_len, 1, 1)
            cums = cums[1:, :, :]
            return pad(cums, p3d, "constant", 0)

    if use_target_fn:

        def get_q_target(o, o_next, r, d):
            a_next = best_action(o, o_next)
            q_target = q_target_copy(torch.cat((o_next, a_next), dim=-1))
            q_target = r + gamma * (1 - d) * q_target
            return q_target

    else:

        def get_q_target(o, o_next, r, d):
            return r

    def update(sample_size=1000, update=True):
        data = buf.sample_episodes(sample_size=sample_size)
        r, o_next, d = data["reward"], data["obs_next"], data["done"]
        a = data["act"]
        o = data["obs"]
        q_optimizer.zero_grad()
        q_estim = q(torch.cat((o, a), dim=-1))
        q_target = get_q_target(o, o_next, r, d)
        q_loss = torch.mean((q_estim - q_target) ** 2)
        if update:
            q_loss.backward()
            q_optimizer.step()
            with torch.no_grad():
                for p, p_target in zip(q.parameters(), q_target_copy.parameters()):
                    p_target.data.mul_(polyak)
                    p_target.data.add_((1 - polyak) * p.data)
        return q_loss

    for i_ep in range(num_ep):
        obs_next = env.reset()
        for i_turn in range(ep_len):
            obs = obs_next
            act = env.action_space.sample()
            obs_next, rwd, done, _ = env.step(act)
            buf.store(obs, act, rwd, obs_next, np.array([done]))
        if (i_ep + 1) % update_every == 0:
            for i_up in range(update_every):
                q_loss = update()
                if i_up == (update_every - 1):
                    if (i_ep + 1) % 200 == 0:
                        print(f"Q loss at {i_ep}: {q_loss}")
    q_loss = update(sample_size=10000)
    return q_loss


def train_LSTMDeterministicActor(
    input_size=1,
    hidden_size=8,
    max_buffer_len=100000,
    pi_lr=1e-3,
    num_ep=10000,
    ep_len=3,
    update_every=10,
    goal_constant=None,
    goal_mean=False,
):
    pi = LSTMDeterministicActor(input_size, hidden_size, action_size=1, low=0, high=1)
    buf = RDPGBuffer(max_buffer_len)
    pi_optimizer = Adam(pi.parameters(), lr=pi_lr)
    if goal_constant is not None:

        def goal(x):
            return goal_constant

    elif not goal_mean:

        def goal(x):
            return x

    else:

        def goal(x):
            return torch.cumsum(x, 0) / torch.range(1, ep_len).view(ep_len, 1, 1)

    def update(sample_size=1000, update=True):
        data = buf.sample_episodes(sample_size=sample_size)
        o = data["obs"]
        pi_optimizer.zero_grad()
        a_proposed = pi(o)
        pi_loss = torch.mean(torch.abs(a_proposed - goal(o)))
        if update:
            pi_loss.backward()
            pi_optimizer.step()
        return pi_loss

    obs_next = np.random.rand(1)
    for i_ep in range(num_ep):
        for i_turn in range(ep_len):
            obs = obs_next
            act = np.random.rand(1)
            rwd = 0.0
            obs_next = np.random.rand(1)
            if i_turn == ep_len - 1:
                done = 1
            else:
                done = 0
            buf.store(obs, act, rwd, obs_next, np.array([done]))
        if (i_ep + 1) % update_every == 0:
            for i_up in range(update_every):
                _ = update()
                if i_up == (update_every - 1):
                    if (i_ep + 1) % 200 == 0:
                        pi_loss = update()
                        print(f"Pi loss at {i_ep}: {pi_loss}")
    pi_loss = update(sample_size=10000)
    return pi_loss


def test_critic_const_target():
    q_loss = train_LSTMEstimator(q_lr=1e-2, num_ep=2000, goal_constant=0.5)
    print(f"Final Q loss, const target: {q_loss}")
    assert q_loss < 0.001


def test_critic_const_use_target_fn():
    q_loss = train_LSTMEstimator(
        q_lr=1e-2, num_ep=2000, goal_constant=0.5, use_target_fn=True
    )
    print(f"Final Q loss, const target: {q_loss}")
    assert q_loss < 0.001


def test_critic_last_target():
    q_loss = train_LSTMEstimator(
        q_lr=1e-2, num_ep=1000, goal_constant=None, goal_mean=False
    )
    print(f"Final Q loss, const target: {q_loss}")
    assert q_loss < 0.001


def test_critic_last_use_target_fn():
    q_loss = train_LSTMEstimator(
        q_lr=1e-2, num_ep=1000, goal_constant=None, goal_mean=False, use_target_fn=True
    )
    print(f"Final Q loss, const target: {q_loss}")
    assert q_loss < 0.001


def test_critic_mean_target():
    q_loss = train_LSTMEstimator(
        q_lr=1e-2, num_ep=1000, goal_constant=None, goal_mean=True
    )
    print(f"Final Q loss, mean obs target: {q_loss}")
    assert q_loss < 0.001


def test_critic_mean_use_target_fn():
    q_loss = train_LSTMEstimator(
        q_lr=2e-3, num_ep=5000, goal_constant=None, goal_mean=True, use_target_fn=True
    )
    print(f"Final Q loss, mean obs target: {q_loss}")
    assert q_loss < 0.001


def test_actor_const_target():
    pi_loss = train_LSTMDeterministicActor(pi_lr=1e-2, num_ep=2000, goal_constant=0.5)
    print(f"Final pi loss, const target: {pi_loss}")
    assert pi_loss < 0.01


def test_actor_last_target():
    pi_loss = train_LSTMDeterministicActor(pi_lr=1e-2, num_ep=2000, goal_mean=False)
    print(f"Final pi loss, last obs target: {pi_loss}")
    assert pi_loss < 0.01


def test_actor_mean_target():
    pi_loss = train_LSTMDeterministicActor(pi_lr=1e-2, num_ep=2000, goal_mean=True)
    print(f"Final pi loss, mean obs target: {pi_loss}")
    assert pi_loss < 0.01
