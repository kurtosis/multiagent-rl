# from multiagent_rl.algos.training import *
from multiagent_rl.algos.ddpg import *
from multiagent_rl.algos.rdpg import *
from multiagent_rl.environments.tournament_env import *

# test_rdpg_buffer()
# test_rdpg_buffer(num_ep=10000, sample_size=1000)
# test_rdpg_buffer(num_ep=100000, sample_size=10000)


# ddpg(
#     seed=1736,
#     epochs=20,
#     steps_per_epoch=500,
#     update_every=100,
#     save_freq=2,
#     start_steps=0,
#     sample_size=128,
#     update_after=0,
#     env_fn=MimicObs,
#     env_kwargs={"reward": "l1", "ep_len": 3, "mean_target": True},
# )

# rdpg(
#     seed=1736,
#     epochs=20,
#     steps_per_epoch=500,
#     update_every=100,
#     save_freq=2,
#     start_steps=0,
#     sample_size=128,
#     update_after=0,
#     env_fn=MimicObs,
#     agent_fn=RDPGAgent,
#     agent_kwargs={"hidden_size": 8},
#     # agent_kwargs={"hidden_size": 8, 'q_fn': 'ContinuousEstimator'},
#     env_kwargs={"reward": "l1", "ep_len": 3, "target": "constant"},
#     # env_kwargs={"reward": "l1", "ep_len": 1, "target": "constant"},
# )

def train_LSTMDeterministicActor(
    input_size=2,
    hidden_size=8,
    max_buffer_len=100000,
    q_lr=1e-3,
    num_ep=10000,
    ep_len=3,
    update_every=10,
    target=0.5,
):
    q = LSTMEstimator(input_size, hidden_size)
    buf = RDPGBuffer(max_buffer_len)
    q_optimizer = Adam(q.parameters(), lr=q_lr)

    def update(sample_size=1000, update=True):
        data = buf.sample_episodes(sample_size=sample_size)
        r, o_next, d = data["reward"], data["obs_next"], data["done"]
        a = data["act"]
        o = data["obs"]
        q_optimizer.zero_grad()
        q_estim = q(torch.cat((o, a), dim=-1))
        q_loss = torch.mean((q_estim - r) ** 2)
        if update:
            q_loss.backward()
            q_optimizer.step()
        return q_loss

    obs_next = np.random.rand(1)
    for i_ep in range(num_ep):
        for i_turn in range(ep_len):
            obs = obs_next
            act = np.random.rand(1)
            rwd = np.abs(act - target)
            obs_next = np.random.rand(1)
            if i_turn == ep_len - 1:
                done = 1
            else:
                done = 0
            buf.store(obs, act, rwd, obs_next, np.array([done]))
        if (i_ep + 1) % update_every == 0:
            # print(f'starting update epoch {i_ep}')
            for i_up in range(update_every):
                q_loss = update()
                # if i_up == (update_every - 1):
                #     if (i_ep + 1) % 1000 == 0:
                #         print(f"update epoch {i_ep}")
                #         print(f'Q loss at {i_ep}: {q_loss}')
    q_loss = update(sample_size=10000)
    return q_loss


# rdpg(
#     seed=1736,
#     epochs=20,
#     steps_per_epoch=1000,
#     # max_episode_len=2,
#     update_every=100,
#     save_freq=2000,
#     start_steps=0,
#     sample_size=1024,
#     update_after=0,
#     env_fn=MimicObs,
#     env_kwargs={"reward": "l1", "ep_len": 3},
# )

# data = buf.sample_episodes(sample_size=1)
# o, a = data["obs"], data["act"]
# r, o_next, d = data["reward"], data["obs_next"], data["done"]
#
# a_offset = 0.
# q_input = torch.cat((o, a + a_offset), dim=-1)
# q = agent.q(q_input)
# print(q)
#
#
# def test_q(o_val, a_val):
#     a_trial = torch.tensor([a_val]*3).reshape_as((a))
#     o_trial = torch.tensor([o_val]*3).reshape_as((o))
#     q_input = torch.cat((o_trial, a_trial), dim=-1)
#     return agent.q(q_input)
#
# o_val = 0.0
# a_val = 0.5
#
# for a_val in np.arange(0.48,0.56,0.01):
# # for a_val in np.arange(0,1.01,0.1):
#     print(a_val)
#     # print(test_q(o_val, a_val))
#     print(test_q(o_val, a_val).squeeze().detach().numpy().tolist())
#
# def test_pi(o_val):
#     o_trial = torch.tensor([o_val]*3).reshape_as((o))
#     return agent.pi(o_trial)
#
# for o_val in np.arange(0,1.01,0.1):
#     print(o_val)
#     print(test_pi(o_val).squeeze().detach().numpy().tolist())

#
# batch_size = r.shape[1]
# with torch.no_grad():
#     a_next = agent_target.pi(o_next)
#     q_target = agent_target.q(torch.cat((o_next, a_next), dim=-1))
#     # reshape so that this will work with DDPG agent (for testing)
#     q_target = q_target.reshape_as(d)
#     q_target = r + gamma * (1 - d) * q_target
