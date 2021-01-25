# import multiagent_rl.algos.orig_ddpg.core as core
# from multiagent_rl.algos.orig_ddpg.ddpg import *

# import multiagent_rl.algos.orig_sac.core as core
# from multiagent_rl.algos.orig_sac.sac import *

# import multiagent_rl.algos.orig_td3.core as core
# from multiagent_rl.algos.orig_td3.td3 import *

# from multiagent_rl.algos.ddpg import *

from multiagent_rl.algos.rdpg import *

# from multiagent_rl.algos.td3 import *
from multiagent_rl.algos.sac import *
from multiagent_rl.algos.rsac import *

# from multiagent_rl.algos.rtd3 import *

from multiagent_rl.environments.tournament_env import *

ep_len = 10
seed = 110
steps_per_epoch = 1000
epochs = 200
pi_lr = 1e-3
q_lr = 1e-3
a_lr = 2e-3
batch_size = 1000
batch_size_eps = 100
start_steps = 3000
update_after = 49
update_every = 50
test_episodes = 10
save_freq = 10
gamma = 0.99

env_kwargs = {
    "ep_len": ep_len,
    "fixed": True,
    "opponent_offer": 0.3,
    "opponent_demand": 0.9,
}

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

n_runs = 5
seed = 1
for i in range(n_runs):
    seed += 1
    rsac(
        seed=seed,
        steps_per_epoch=200,
        epochs=epochs,
        pi_lr=pi_lr,
        q_lr=q_lr,
        a_lr=a_lr,
        batch_size=100,
        start_steps=start_steps,
        update_after=update_after,
        update_every=update_every,
        save_freq=save_freq,
        num_test_episodes=100,
        max_episode_len=ep_len,
        env_fn=DistribDualUltimatum,
        env_kwargs={"ep_len": ep_len, "fixed": False},
        agent_fn=RSACAgent,
        agent_kwargs={
            "hidden_size_pi": 8,
            "hidden_size_q": 8,
            "mlp_layers_pi": (32, 32),
            "mlp_layers_q": (32, 32, 32),
        },
        gamma=0.99,
        alpha=0.05,
        update_alpha_after=15000,
        target_entropy=-8.0,
        logger_kwargs={
            "output_dir": "~/research/multiagent-rl/data/testing/constantbot/rsac/",
            "exp_name": "distrib_2",
        },
        q_filename="~/research/multiagent-rl/data/q_maps/distrib_2",
        save_q_every=5000,
    )

    # sac(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     lr=pi_lr,
    #     a_lr = 2e-3,
    #     batch_size=batch_size,
    #     start_steps=start_steps,
    #     update_after=update_after,
    #     update_every=update_every,
    #     save_freq=save_freq,
    #     actor_critic=core.MLPActorCritic,
    #     num_test_episodes=test_episodes,
    #     max_ep_len=ep_len,
    #     gamma=gamma,
    #     ac_kwargs={"hidden_sizes": (32, 32)},
    #     env_fn=ConstantDualUltimatum,
    #     logger_kwargs={
    #         "output_dir": "~/research/multiagent-rl/data/testing/constantbot/sac_orig/",
    #         "exp_name": "done_auto_alpha_targ_4",
    #     },
    #     alpha=0.05,
    # )

    # sac_new(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     a_lr=2e-3,
    #     batch_size=batch_size,
    #     start_steps=start_steps,
    #     update_after=update_after,
    #     update_every=update_every,
    #     save_freq=save_freq,
    #     num_test_episodes=test_episodes,
    #     max_episode_len=ep_len,
    #     agent_kwargs={"hidden_layers_pi": (32, 32), "hidden_layers_q": (32, 32)},
    #     env_fn=ConstantDualUltimatum,
    #     logger_kwargs={
    #         "output_dir": "~/research/multiagent-rl/data/testing/constantbot/sac/",
    #         "exp_name": "gamma_50_alpha_05_targ_4",
    #     },
    #     gamma=0.5,
    #     alpha=0.05,
    #     target_entropy=-4.0,
    # )
    #
    # sac_new(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     a_lr = 2e-3,
    #     batch_size=batch_size,
    #     start_steps=start_steps,
    #     update_after=update_after,
    #     update_every=update_every,
    #     save_freq=save_freq,
    #     num_test_episodes=test_episodes,
    #     max_episode_len=ep_len,
    #     agent_kwargs={"hidden_layers_pi": (32, 32), "hidden_layers_q": (32, 32)},
    #     env_fn=ConstantDualUltimatum,
    #     logger_kwargs={
    #         "output_dir": "~/research/multiagent-rl/data/testing/constantbot/sac/",
    #         "exp_name": "gamma_90_alpha_05_targ_4",
    #     },
    #     gamma=0.9,
    #     alpha=0.05,
    #     target_entropy = -4.0,
    # )

    # sac(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     a_lr=a_lr,
    #     batch_size=batch_size,
    #     start_steps=start_steps,
    #     update_after=update_after,
    #     update_every=update_every,
    #     save_freq=save_freq,
    #     num_test_episodes=test_episodes,
    #     max_episode_len=ep_len,
    #     agent_kwargs={"hidden_layers_pi": (32, 32), "hidden_layers_q": (32, 32)},
    #     env_fn=ConstantDualUltimatum,
    #     logger_kwargs={
    #         "output_dir": "~/research/multiagent-rl/data/testing/constantbot/sac/",
    #         "exp_name": "alpha_8_new_buf",
    #     },
    #     gamma=0.99,
    #     alpha=0.05,
    #     update_alpha_after=15000,
    #     target_entropy=-8.0,
    #     q_filename="~/research/multiagent-rl/data/q_alpha_8",
    #     save_q_every=5000,
    # )

    # ddpg(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=300,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     batch_size=batch_size,
    #     start_steps=10000,
    #     update_after=update_after,
    #     update_every=update_every,
    #     save_freq=save_freq,
    #     actor_critic=core.MLPActorCritic,
    #     num_test_episodes=test_episodes,
    #     max_ep_len=ep_len,
    #     gamma=0,
    #     # act_noise=0.01,
    #     ac_kwargs={"hidden_sizes": (32, 32, 32, 32)},
    #     env_fn=ConstantDualUltimatum,
    #     # env_kwargs=env_kwargs,
    #     logger_kwargs={
    #         # "output_dir": "~/research/multiagent-rl/data/testing/constantbot/ddpg_orig/",
    #         # "exp_name": "long_start",
    #     },
    # )

    # ddpg_new(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     batch_size=batch_size,
    #     start_steps=start_steps,
    #     update_after=update_after,
    #     update_every=update_every,
    #     save_freq=save_freq,
    #     agent_fn=DDPGAgent,
    #     num_test_episodes=test_episodes,
    #     max_episode_len=ep_len,
    #     gamma=gamma,
    #     agent_kwargs={"hidden_layers_mu": (4, 4), "hidden_layers_q": (32, 32, 32, 32), "noise_std": 0.5},
    #     env_fn=ConstantDualUltimatum,
    #     env_kwargs=env_kwargs,
    #     logger_kwargs={
    #         "output_dir": "~/research/multiagent-rl/data/testing/constantbot/ddpg/",
    #         "exp_name": "temp",
    #     },
    # )
    #
    # rdpg(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     batch_size=batch_size_eps,
    #     start_steps=1,
    #     update_after=update_after,
    #     update_every=update_every,
    #     test_episodes=test_episodes,
    #     max_episode_len=ep_len,
    #     save_freq=save_freq,
    #     agent_fn=RDPGAgent,
    #     gamma=gamma,
    #     agent_kwargs={"hidden_layers_mu": (4, 4), "hidden_layers_q": (32, 32, 32, 32)},
    #     env_fn=ConstantDualUltimatum,
    #     env_kwargs=env_kwargs,
    #     logger_kwargs={
    #         "output_dir": "~/research/multiagent-rl/data/testing/constantbot/rdpg/",
    #         "exp_name": "temp",
    #     },
    # )

    # td3(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     batch_size=batch_size,
    #     start_steps=start_steps,
    #     update_after=update_after,
    #     update_every=update_every,
    #     save_freq=save_freq,
    #     actor_critic=core.MLPActorCritic,
    #     num_test_episodes=test_episodes,
    #     max_ep_len=ep_len,
    #     gamma=gamma,
    #     ac_kwargs={"hidden_sizes": (32, 32,)},
    #     env_fn=ConstantDualUltimatum,
    #     logger_kwargs={
    #         "output_dir": "~/research/multiagent-rl/data/testing/constantbot/td3_orig/",
    #         "exp_name": "non_flat_reward",
    #     },
    # )

    # td3_new(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     batch_size=batch_size,
    #     start_steps=start_steps,
    #     update_after=update_after,
    #     update_every=update_every,
    #     save_freq=save_freq,
    #     agent_fn=TD3Agent,
    #     num_test_episodes=test_episodes,
    #     max_episode_len=ep_len,
    #     gamma=gamma,
    #     agent_kwargs={"hidden_layers_mu": (32, 32), "hidden_layers_q": (32, 32)},
    #     env_fn=ConstantDualUltimatum,
    #     logger_kwargs={"output_dir" : "~/research/multiagent-rl/data/testing/constantbot/td3/", "exp_name": "nonflat_reward"},
    #     env_kwargs={
    #         "ep_len": ep_len,
    #         "fixed": True,
    #         "opponent_offer": 0.3,
    #         "opponent_demand": 0.9,
    #         "reward": "non_flat",
    #     }
    # )

    # rtd3(
    #     seed=seed,
    #     steps_per_epoch=steps_per_epoch,
    #     epochs=epochs,
    #     pi_lr=pi_lr,
    #     q_lr=q_lr,
    #     batch_size=batch_size_eps,
    #     start_steps=start_steps,
    #     update_after=update_after,
    #     update_every=update_every,
    #     test_episodes=test_episodes,
    #     max_episode_len=ep_len,
    #     save_freq=save_freq,
    #     agent_fn=TD3Agent,
    #     gamma=gamma,
    #     agent_kwargs={"hidden_layers_mu": (256, 256), "hidden_layers_q": (256, 256)},
    #     env_fn=ConstantDualUltimatum,
    #     env_kwargs=env_kwargs,
    #     logger_kwargs={"output_dir" : "~/research/multiagent-rl/data/testing/constantbot/rtd3/"}
    # )


# ddpg(
#     ConstantDualUltimatum,
#     seed=0,
#     steps_per_epoch=1000,
#     epochs=200,
#     pi_lr=1e-3,
#     q_lr=1e-3,
#     batch_size=1024,
#     start_steps=3000,
#     update_after=949,
#     update_every=50,
#     act_noise=0.1,
#     num_test_episodes=10,
#     max_ep_len=ep_len,
#     logger_kwargs=dict(),
#     save_freq=10,
#     gamma=0.,
# )

# td3(
#     ConstantDualUltimatum,
#     seed=0,
#     steps_per_epoch=1000,
#     epochs=200,
#     pi_lr=1e-3,
#     q_lr=1e-3,
#     batch_size=1024,
#     start_steps=3000,
#     update_after=949,
#     update_every=50,
#     act_noise=0.1,
#     num_test_episodes=10,
#     max_ep_len=ep_len,
#     save_freq=10,
# )

# td3_new(
#     ConstantDualUltimatum,
#     env_kwargs={
#         "ep_len": ep_len,
#         # "fixed": False,
#         # "opponent_offer": 0.5,
#         # "opponent_demand": 0.5,
#     },
#     seed=0,
#     steps_per_epoch=1000,
#     epochs=200,
#     pi_lr=1e-3,
#     q_lr=1e-3,
#     batch_size=1024,
#     start_steps=3000,
#     update_after=949,
#     update_every=50,
#     num_test_episodes=10,
#     max_episode_len=ep_len,
#     save_freq=10,
# )


# rtd3(
#     seed=seed,
#     steps_per_epoch=steps_per_epoch,
#     epochs=epochs,
#     pi_lr=pi_lr,
#     q_lr=q_lr,
#     batch_size=batch_size,
#     start_steps=start_steps,
#     update_after=update_after,
#     update_every=update_every,
#     test_episodes=test_episodes,
#     max_episode_len=ep_len,
#     save_freq=save_freq,
#     agent_fn=TD3Agent,
#     env_fn=ConstantDualUltimatum,
# )
