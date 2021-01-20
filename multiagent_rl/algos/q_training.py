"""
These functions are for training a Q function only, without pi.
Actions are chosen either randomly or else from known best value
"""
import os

from multiagent_rl.algos.agents import *
from multiagent_rl.algos.training import count_vars
from multiagent_rl.utils.logx import EpochLogger
from multiagent_rl.algos.buffers import *
from multiagent_rl.algos.orig_ddpg.ddpg import ReplayBuffer
from multiagent_rl.utils.evaluation_utils import *
from multiagent_rl.environments.tournament_env import *

max_ep_len = 10
q_lr = 1e-3
update_after = 0
update_every = 200
save_q_every = 1000
q_filename = "/Users/kurtsmith/q_maps_32x4.csv"
batch_size = 1000
activation = nn.ReLU
final_activation = nn.Tanh
steps_per_epoch = 1000
epochs = 20
gamma = 0.99
polyak = 0.995
replay_size = 100000

env = ConstantDualUltimatum()
test_env = deepcopy(env)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

hidden_layers_q = (32, 32, 32, 32)
layer_sizes_q = [obs_dim + act_dim] + list(hidden_layers_q) + [1]

q = ContinuousEstimator(layer_sizes=layer_sizes_q, activation=activation)
q_optimizer = Adam(q.parameters(), lr=q_lr)
q_target = deepcopy(q)
replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)


def compute_loss_q(data):
    o, a, r, o2, d = (
        data["obs"],
        data["act"],
        data["rew"],
        data["obs2"],
        data["done"],
    )

    q_est = q(torch.cat([o, a], dim=-1))
    EPS = 1e-6
    # Bellman backup for Q function
    with torch.no_grad():
        a2 = torch.tensor([env.opponent_demand + EPS, env.opponent_offer - EPS])
        a2 = a2.repeat(o2.shape[0], 1)

        q_pi_targ = q_target(torch.cat([o2, a2], dim=-1))
        backup = r + gamma * (1 - d) * q_pi_targ

    # MSE loss against Bellman backup
    loss_q = ((q_est - backup) ** 2).mean()

    # Useful info for logging
    loss_info = dict(QVals=q_est.detach().numpy())

    return loss_q, loss_info


def update(data):
    # First run one gradient descent step for Q.
    q_optimizer.zero_grad()
    loss_q, loss_info = compute_loss_q(data)
    loss_q.backward()
    q_optimizer.step()
    # Record things
    # logger.store(LossQ=loss_q.item(), LossPi=loss_pi.item(), **loss_info)

    # Finally, update target networks by polyak averaging.
    with torch.no_grad():
        for p, p_targ in zip(q.parameters(), q_target.parameters()):
            p_targ.data.mul_(polyak)
            p_targ.data.add_((1 - polyak) * p.data)


def save_q_map(q, filename, step, n=101):
    grid = np.linspace(0, 1, n)
    a0, a1 = np.meshgrid(grid, grid)
    a0 = a0.reshape(-1)
    a1 = a1.reshape(-1)
    a = torch.tensor(np.stack((a0, a1), axis=1), dtype=torch.float32)
    o = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
    o = o.repeat(a.shape[0], 1)
    with torch.no_grad():
        q_estim = q(torch.cat([o, a], dim=-1))
    result = torch.cat([o, a, q_estim.unsqueeze(dim=1)], dim=1)
    cols = (
        [f"o_{i}" for i in range(o.shape[-1])]
        + [f"a_{i}" for i in range(a.shape[-1])]
        + ["q"]
    )
    result = pd.DataFrame(result.numpy(), columns=cols)
    result["step"] = step
    if not os.path.isfile(filename):
        result.to_csv(filename, index=False, header="column_names")
    else:
        result.to_csv(filename, index=False, mode="a", header=False)


def main():
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    ep_ret = 0
    ep_len = 0
    for t in range(total_steps):
        a = env.action_space.sample()
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for _ in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)
        if t % save_q_every == 0:
            save_q_map(q, q_filename, step=t)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch
            o, ep_ret, ep_len = env.reset(), 0, 0


if __name__ == "__main__":
    main()
