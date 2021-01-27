import numpy as np

# import os, sys
# from multiagent_rl.environments.tournament_env import *
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
#
# sys.path.insert(
#     0, "/Users/kurtsmith/research/pytorch_projects/reinforcement_learning/environments"
# )
# sys.path.insert(0, "/Users/kurtsmith/research/spinningup")


class Agent:
    """
    Top level Agent class for Dual Ultimatum game.
    """

    # def __init__(
    #     self,
    #     *args,
    #     **kwargs,
    # ):
    #     pass

    def act(self):
        raise NotImplementedError

    def reset_state(self, *args, **kwargs):
        pass

    def update(self, *args, **kwargs):
        pass

    def store_to_buffer(self, *args, **kwargs):
        pass


class ConstantBot(Agent):
    """
    Static bot that plays a constant action in Dual Ultimatum episode.
    fixed flag controls allows the action to be reset at the start of a new episode.

    """

    def __init__(
        self,
        *args,
        offer=None,
        demand=None,
        mean_offer=None,
        std_offer=None,
        mean_demand=None,
        std_demand=None,
        fixed=False,
        **kwargs,
    ):
        super().__init__()
        self.fixed = fixed
        self.mean_offer = mean_offer
        self.std_offer = std_offer
        self.mean_demand = mean_demand
        self.std_demand = std_demand

        self.offer = self.set_action(offer, self.mean_offer, self.std_offer)
        self.demand = self.set_action(demand, self.mean_demand, self.std_demand)

    def set_action(self, value, mean, std):
        if value is not None:
            return value
        elif mean is not None and std is not None:
            return (1 + np.tanh(mean + std * np.random.randn(1)[0])) / 2
        else:
            return np.random.rand(1)[0]

    def act(self, *args, **kwargs):
        return np.array((self.offer, self.demand))

    def reset_state(self, *args, **kwargs):
        if not self.fixed:
            self.offer = self.set_action(None, self.mean_offer, self.std_offer)
            self.demand = self.set_action(None, self.mean_demand, self.std_demand)


class DistribBot(Agent):
    """
    Bot that plays a draw from a static distribution, based on tanh transform.
    To do: Could implement this using beta or log-odds normal distr instead, easier to reason about?
    """

    def __init__(
        self,
        *args,
        mean_offer=None,
        std_offer=None,
        mean_demand=None,
        std_demand=None,
        **kwargs,
    ):
        super().__init__()
        # Initialized with approximate mean/std values (in (0,1)) for simplicity.
        # Note these aren't exact means b/c of the nonlinear tanh transform.

        # Sample mean from uniform dist
        if mean_offer is None:
            mean_offer = np.random.rand()
        if mean_demand is None:
            mean_demand = np.random.rand()
        if std_offer is None:
            std_offer = np.random.rand()
        if std_demand is None:
            std_demand = np.random.rand()

        self.approx_mean_offer = mean_offer
        self.mean_tanh_offer = np.arctanh(2 * mean_offer - 1)
        self.std_offer = std_offer
        self.approx_mean_demand = mean_demand
        self.mean_tanh_demand = np.arctanh(2 * mean_demand - 1)
        self.std_demand = std_demand

    def act(self, *args, **kwargs):
        offer = (
            1 + np.tanh(self.mean_tanh_offer + self.std_offer * np.random.randn(1)[0])
        ) / 2
        demand = (
            1 + np.tanh(self.mean_tanh_demand + self.std_demand * np.random.randn(1)[0])
        ) / 2
        return np.array((offer, demand))


class MimicBot(Agent):
    def __init__(self):
        super().__init__()

    def act(self, last_offer=None, last_demand=None):
        if last_offer is None:
            last_offer = np.random.rand(1)
        if last_demand is None:
            last_demand = np.random.rand(1)
        return np.array((last_offer, last_demand))


class BenchmarkBot(Agent):
    def __init__(self, benchmark=1):
        super().__init__()
        self.benchmark = benchmark
        self.cum_total = 0


class GreedFearBot(Agent):
    def __init__(self, greed=1, fear=1):
        super().__init__()
        self.greed = greed
        self.fear = fear

    def act(self, last_offer, last_demand, accepted):
        if accepted:
            offer = last_offer - 0
            demand = last_demand + 0
        else:
            offer = last_offer + 0
            demand = last_demand - 0
        return np.array((offer, demand))
