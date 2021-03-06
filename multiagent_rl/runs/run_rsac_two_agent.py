from multiagent_rl.agents.bot_agents import *
from multiagent_rl.agents.agents import *
from multiagent_rl.environments import *
from multiagent_rl.trainers.train_rsac_two_agent import train_rsac_two_agent

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


# agent_fns=[RSACAgent, DistribBot]
# agent_kwargs = [rsac_agent_kwargs, {}]

# agent_fns = [RSACAgent, ConstantBot]
# agent_kwargs = [rsac_agent_kwargs, constantbot_kwargs]

# agent_fns=[DistribBot, DistribBot]
# agent_kwargs = [{}, {}]

agent_fns = [RSACAgent, RSACAgent]
agent_kwargs = [rsac_agent_kwargs, rsac_agent_kwargs]

train_rsac_two_agent(
    env_fn=DualUltimatum,
    env_kwargs=dict(),
    agent_fns=agent_fns,
    agent_kwargs=agent_kwargs,
    seed=0,
    steps_per_epoch=100,
    epochs=10,
    batch_size=100,
    start_steps=100,
    update_after=10,
    update_every=50,
    num_test_episodes=10,
    max_ep_len=10,
    logger_kwargs=dict(),
    save_freq=1,
)
