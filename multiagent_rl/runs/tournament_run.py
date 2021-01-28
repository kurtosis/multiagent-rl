from textwrap import dedent

from multiagent_rl.trainers.tournament import *
from multiagent_rl.agents.agents import *
from multiagent_rl.agents.bot_agents import *
from multiagent_rl.environments.tournament_env import *

from multiagent_rl.user_config import DEFAULT_DATA_DIR
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


agent_fns = [ConstantBot] * 4
agent_kwargs = [
    {"offer": 0.2, "demand": 0.4, "fixed": True},
    {"offer": 0.9, "demand": 0.1, "fixed": True},
    {"offer": 0.5, "demand": 0.5, "fixed": True},
    {"offer": 0.1, "demand": 0.8, "fixed": True},
]


# Test DDPG vs constant bots
logger_kwargs, output_msg = logging_info("tournament", subdir="test")
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
)


train_tournament(
    env_fn=RoundRobinTournament,
    env_kwargs=env_kwargs,
    agent_fns=agent_fns,
    agent_kwargs=agent_kwargs,
    seed=0,
    start_steps=100,
    update_after=0,
    logger_kwargs=logger_kwargs,
)
