from textwrap import dedent

from multiagent_rl.algos.tournament_ddpg import *
from multiagent_rl.algos.ultimatum_agents import *
from multiagent_rl.environments.tournament_env import *

from multiagent_rl.user_config import (
    DEFAULT_DATA_DIR,
)
from multiagent_rl.utils.logx import colorize

DIV_LINE_WIDTH = 80


def create_output_msg(logger_kwargs):
    plot_cmd = "python -m multiagent_rl.run plot " + logger_kwargs["output_dir"]
    plot_cmd = colorize(plot_cmd, "green")
    test_cmd = "python -m multiagent_rl.run test_policy " + logger_kwargs["output_dir"]
    test_cmd = colorize(test_cmd, "green")
    output_msg = (
        "\n" * 5
        + "=" * DIV_LINE_WIDTH
        + "\n"
        + dedent(
            """\
End of experiment.


Plot results from this run with:

%s


Watch the trained agent with:

%s


"""
            % (plot_cmd, test_cmd)
        )
        + "=" * DIV_LINE_WIDTH
        + "\n" * 5
    )
    return output_msg


def logging_info(env_str, subdir=None):
    env_dir = env_str.lower().replace("-", "_")
    if subdir is None:
        output_dir = f"{DEFAULT_DATA_DIR}/{env_dir}/{int(time.time())}"
    else:
        output_dir = f"{DEFAULT_DATA_DIR}/{env_dir}/{subdir}/{int(time.time())}"
    logger_kwargs = {"output_dir": output_dir}
    output_msg = create_output_msg(logger_kwargs)
    return logger_kwargs, output_msg


seed = 4153
epochs = 10
steps_per_epoch = 4000
runs_per_method = 4
runs_per_method_ppo = 4
max_episode_len = 100

logger_kwargs, output_msg = logging_info("dual_ultimatum", subdir="dual_ultimatum")

static_agent_kwargs = dict(
    mean_offer=0.2, std_offer=0.00001, mean_threshold=0.55, std_threshold=0.00001
)

# Test all bots
# logger_kwargs, output_msg = logging_info("tournament", subdir="dual_ultimatum")
# env_kwargs = dict(top_cutoff=2, bottom_cutoff=None, top_reward=1.0, bottom_reward=1.0,)
# agent_kwargs_constantbot = dict(offer=0.6, threshold=0.2)
# agent_kwargs_distrib = dict(
#     mean_offer=0.3, std_offer=0.01, mean_threshold=0.9, std_threshold=0.01,
# )
# tournament_ddpg(
#     seed=42,
#     steps_per_epoch=4000,
#     num_agents=4,
#     env_fn=RoundRobinTournament,
#     # agent_fn=ConstantBot,
#     # agent_kwargs=agent_kwargs_constantbot,
#     # agent_fn=StaticDistribBot,
#     # agent_kwargs=agent_kwargs_distrib,
#     logger_kwargs=logger_kwargs,
#     env_kwargs=env_kwargs,
# )


# Test DDPG vs constant bots
logger_kwargs, output_msg = logging_info("tournament", subdir="dual_ultimatum")
env_kwargs = dict(
    num_rounds=2,
    round_length=3,
    noise_size=0,
    top_cutoff=1,
    bottom_cutoff=None,
    top_reward=1.0,
    bottom_reward=1.0,
    score_reward=False,
    per_turn_reward=False,
    hide_obs=False,
    hide_score=False,
    # game_kwargs=dict(reward="l1_const"),
)
agent_kwargs_constantbot = dict(offer=0.5, threshold=0.3)
agent_kwargs_staticdistribbot = dict(
    mean_offer=0.6, std_offer=0., mean_threshold=0.4, std_threshold=0.,
)

agent_kwargs_ddpg = dict(
    hidden_layers_mu=(1,),
    hidden_layers_q=(64, 64, 64),
    # hidden_layers_q=(64, 64, 64, 64),
    noise_std=0.3,
    pi_lr=1e-3,
    q_lr=1e-3,
    # gamma=0,
)


agent_kwargs_lstm = dict(
)
# agents = [DDPGAgent, ConstantBot, ConstantBot, ConstantBot]
# agents_kwargs = [
#     agent_kwargs_ddpg,
#     agent_kwargs_constantbot,
#     agent_kwargs_constantbot,
#     agent_kwargs_constantbot,
# ]
agents = [DDPGLSTMAgent, StaticDistribBot, StaticDistribBot, StaticDistribBot]
agents_kwargs = [
    # agent_kwargs_ddpg,
    agent_kwargs_lstm,
    agent_kwargs_staticdistribbot,
    agent_kwargs_staticdistribbot,
    agent_kwargs_staticdistribbot,
]

tournament_ddpg(
    seed=1736,
    steps_per_epoch=5000,
    epochs=50,
    save_freq=2,
    start_steps=0,
    sample_size=4096,
    update_after=0,
    env_fn=RoundRobinTournament,
    agents=agents,
    agents_kwargs=agents_kwargs,
    logger_kwargs=logger_kwargs,
    env_kwargs=env_kwargs,
    q_file="q",
)

