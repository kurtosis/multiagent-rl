from multiagent_rl.algos.ddpg import *
from multiagent_rl.algos.rdpg import *
from multiagent_rl.environments.tournament_env import *



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

ep_len=2
rdpg(
    epochs=20,
    steps_per_epoch=600,
    update_every=10,
    save_freq=2000,
    start_steps=1000,
    sample_size=1024,
    # sample_size=3072,
    update_after=0,
    env_fn=MimicObs,
    agent_fn=RDPGAgent,
    pi_lr=1e-2,
    q_lr=1e-2,
    max_episode_len=ep_len,
    agent_kwargs={"hidden_size": 16},
    env_kwargs={"reward": "l1", "ep_len": ep_len, "goal_constant": 0.5},
    # env_kwargs={"reward": "l1", "ep_len": 3, "goal_mean": False},
    # env_kwargs={"reward": "l1", "ep_len": 2, "goal_mean": True},
)

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
