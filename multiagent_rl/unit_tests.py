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

rdpg(
    seed=1736,
    epochs=20,
    steps_per_epoch=500,
    update_every=100,
    save_freq=2,
    start_steps=0,
    sample_size=128,
    update_after=0,
    env_fn=MimicObs,
    agent_fn=DDPGAgent,
    agent_kwargs={"hidden_size": 8},
    env_kwargs={"reward": "l1", "ep_len": 3, "target": "constant"},
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


