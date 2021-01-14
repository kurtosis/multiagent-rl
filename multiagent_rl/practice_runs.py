import multiagent_rl.algos.orig_ddpg.core as core
from multiagent_rl.algos.orig_ddpg.ddpg import *
from multiagent_rl.algos.ddpg import *
# from multiagent_rl.algos.rdpg import *
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
# ddpg_new(
#     seed=0,
#     steps_per_epoch=1000,
#     epochs=20,
#     pi_lr = 1e-3,
#     q_lr = 1e-3,
#     batch_size=1024,
#     start_steps=20000,
#     update_after=949,
#     update_every=50,
#     save_freq=1,
#     agent_fn=DDPGAgent,
#     # agent_fn=core.MLPActorCritic,
#     num_test_episodes=10,
#     max_episode_len=ep_len,
#     agent_kwargs={"hidden_layers_mu": (256, 256), "hidden_layers_q": (256, 256)},
#     env_fn=ConstantDualUltimatum,
#     env_kwargs={
#         "ep_len": ep_len,
#         "fixed": True,
#         "opponent_offer": 0.5,
#         "opponent_demand": 0.5,
#     },
# )

# ddpg(
#     ConstantDualUltimatum,
#     seed=0,
#     steps_per_epoch=1000,
#     epochs=20,
#     pi_lr=1e-3,
#     q_lr=1e-3,
#     batch_size=1024,
#     start_steps=20000,
#     update_after=949,
#     update_every=50,
#     act_noise=0.1,
#     num_test_episodes=10,
#     max_ep_len=ep_len,
#     logger_kwargs=dict(),
#     save_freq=1,
# )

ddpg_new(
    ConstantDualUltimatum,
    seed=0,
    steps_per_epoch=100,
    epochs=2,
    pi_lr=1e-3,
    q_lr=1e-3,
    batch_size=64,
    start_steps=100,
    update_after=49,
    update_every=50,
    num_test_episodes=10,
    max_episode_len=ep_len,
    # act_noise=0.1,
    # max_ep_len=ep_len,
    logger_kwargs=dict(),
    save_freq=1,
)

ddpg(
    ConstantDualUltimatum,
    seed=0,
    steps_per_epoch=100,
    epochs=2,
    pi_lr=1e-3,
    q_lr=1e-3,
    batch_size=64,
    start_steps=100,
    update_after=49,
    update_every=50,
    num_test_episodes=10,
    # max_episode_len=ep_len,
    act_noise=0.1,
    max_ep_len=ep_len,
    logger_kwargs=dict(),
    save_freq=1,
)