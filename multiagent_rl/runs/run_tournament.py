from textwrap import dedent
import time

from multiagent_rl.agents.agents import *
from multiagent_rl.agents.bot_agents import *
from multiagent_rl.environments import *
from multiagent_rl.trainers.train_tournament import train_tournament

train_tournament(agent_fns=[DistribBot, DistribBot, DistribBot, DistribBot])
