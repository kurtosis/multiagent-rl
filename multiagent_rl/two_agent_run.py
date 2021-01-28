from multiagent_rl.algos.bot_agents import *
from multiagent_rl.algos.rsac_two_agent import *


from multiagent_rl.environments.tournament_env import *

# agent_fns=[RSACAgent, DistribBot]
agent_fns = [RSACAgent, ConstantBot]
rsac_agent_kwargs = {
    "hidden_size_pi": 8,
    "hidden_size_q": 8,
    "mlp_layers_pi": (32, 32),
    "mlp_layers_q": (32, 32, 32),
    "alpha": 0.05,
    "update_alpha_after": 100,
    "target_entropy": -4.0,
}

constantbot_kwargs = {"offer": 0.2, "demand": 0.4, "fixed": True}

agent_kwargs = [rsac_agent_kwargs, constantbot_kwargs]

# agent_fns=[DistribBot, DistribBot]
# agent_kwargs = [{}, {}]

two_agent_rsac(
    env_fn=DualUltimatum,
    env_kwargs=dict(),
    agent_fns=agent_fns,
    agent_kwargs=agent_kwargs,
    seed=0,
    steps_per_epoch=400,
    epochs=100,
    batch_size=100,
    start_steps=100,
    update_after=10,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=10,
    logger_kwargs=dict(),
    save_freq=1,
)
