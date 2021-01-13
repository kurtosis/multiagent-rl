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

# ep_len = 10
# rdpg(
#     epochs=20,
#     steps_per_epoch=600,
#     update_every=10,
#     save_freq=2000,
#     start_steps=1000,
#     sample_size=1024,
#     update_after=0,
#     agent_fn=RDPGAgent,
#     pi_lr=1e-2,
#     q_lr=1e-2,
#     max_episode_len=ep_len,
#     agent_kwargs={"hidden_size": 16},
#     # env_fn=MimicObs,
#     # env_kwargs={"reward": "l1", "ep_len": ep_len, "goal_constant": 0.5},
#     # env_kwargs={"reward": "l1", "ep_len": ep_len, "goal_mean": False},
#     # env_kwargs={"reward": "l1", "ep_len": ep_len, "goal_mean": True},
#     env_fn=ConstantDualUltimatum,
#     env_kwargs={"ep_len": ep_len, "fixed": True, "opponent_offer": 0.5, "opponent_demand": 0.5},
# )

ep_len = 10
ddpg(
    epochs=20,
    steps_per_epoch=600,
    update_every=10,
    save_freq=2000,
    start_steps=2000,
    sample_size=1024,
    update_after=0,
    agent_fn=DDPGAgent,
    pi_lr=2e-3,
    q_lr=2e-3,
    max_episode_len=ep_len,
    agent_kwargs={"hidden_layers_mu": (64, 64, 64), "hidden_layers_q": (64, 64, 64)},
    env_fn=ConstantDualUltimatum,
    env_kwargs={
        "ep_len": ep_len,
        "fixed": True,
        "opponent_offer": 0.5,
        "opponent_demand": 0.5,
    },
)
