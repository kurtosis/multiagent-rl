from multiagent_rl.agents.agents import RSACAgent
from multiagent_rl.environments import *
from multiagent_rl.trainers.train_rsac import train_rsac

n_runs = 1
seed = 1
for i in range(n_runs):
    seed += 1
    train_rsac(
        seed=seed,
        steps_per_epoch=100,
        epochs=4,
        pi_lr=1e-3,
        q_lr=1e-3,
        a_lr=1e-3,
        batch_size=100,
        start_steps=3000,
        update_after=49,
        update_every=50,
        save_freq=10,
        num_test_episodes=100,
        max_episode_len=10,
        env_fn=DistribDualUltimatum,
        env_kwargs={"ep_len": 10, "fixed": False},
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
            # "output_dir": "~/research/multiagent-rl/data/testing/constantbot/rsac/",
            # "exp_name": "distrib_2",
        },
    )
