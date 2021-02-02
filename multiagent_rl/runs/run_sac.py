from multiagent_rl.agents.agents import SACAgent
from multiagent_rl.environments import *
from multiagent_rl.trainers.train_sac import train_sac
from multiagent_rl.trainers.train_rsac import train_rsac

n_runs = 5
seed = 1
for i in range(n_runs):
    seed += 1

    train_sac(
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
        num_test_episodes=10,
        max_episode_len=10,
        agent_kwargs={"hidden_layers_pi": (32, 32), "hidden_layers_q": (32, 32)},
        env_fn=ConstantDualUltimatum,
        env_kwargs={"ep_len": 10, "fixed": False},
        gamma=0.5,
        alpha=0.05,
        target_entropy=-4.0,
    )
