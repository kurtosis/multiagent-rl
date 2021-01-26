
from multiagent_rl.algos.bot_agents import *
from multiagent_rl.algos.rsac_two_agent import *


from multiagent_rl.environments.tournament_env import *

agent_fns=[RSACAgent, DistribBot]
rsac_agent_kwargs = {
                   "hidden_size_pi": 8,
                   "hidden_size_q": 8,
                   "mlp_layers_pi": (32, 32),
                   "mlp_layers_q": (32, 32, 32),
               }

agent_kwargs = [rsac_agent_kwargs, {}]

two_agent_rsac(
    env_fn=DualUltimatum,
    env_kwargs=dict(),
    agent_fns=agent_fns,
    agent_kwargs=agent_kwargs,
    seed=0,
    steps_per_epoch=4000,
    epochs=100,
    max_buffer_len=100000,
    gamma=0.99,
    polyak=0.995,
    pi_lr=1e-3,
    q_lr=1e-3,
    a_lr=1e-3,
    batch_size=100,
    start_steps=10000,
    update_after=1000,
    update_every=50,
    num_test_episodes=10,
    log_interval=10,
    max_episode_len=10,
    logger_kwargs=dict(),
    save_freq=1,
    alpha=0.05,
    update_alpha_after=5000,
    target_entropy=-4.0,
    save_q_every=0,
    q_filename="/Users/kurtsmith/research/multiagent-rl/data/q",
)