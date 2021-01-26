# import os, sys
# from multiagent_rl.environments.tournament_env import *
#
# os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
#
# sys.path.insert(
#     0, "/Users/kurtsmith/research/pytorch_projects/reinforcement_learning/environments"
# )
# sys.path.insert(0, "/Users/kurtsmith/research/spinningup")

class ConstantBot:
    """
    Static bot that plays a constant action in Dual Ultimatum game.
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
        **kwargs,
    ):
        def set_action(value, mean, std):
            if value is not None:
                return value
            elif mean is not None and std is not None:
                return (1 + np.tanh(mean_offer + std_offer * np.random.randn(1)[0])) / 2
            else:
                return np.random.rand(1)[0]

        self.offer = set_action(offer, mean_offer, std_offer)
        self.demand = set_action(demand, mean_demand, std_demand)

    def act(self, *args, **kwargs):
        return np.array((self.offer, self.demand))

    def update(self, *args, **kwargs):
        pass


class DistribBot:
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

    def act(self):
        offer = (
            1 + np.tanh(self.mean_tanh_offer + self.std_offer * np.random.randn(1)[0])
        ) / 2
        demand = (
            1 + np.tanh(self.mean_tanh_demand + self.std_demand * np.random.randn(1)[0])
        ) / 2
        return np.array((offer, demand))

    def update(self, *args, **kwargs):
        pass


class MimicBot:
    def __init__(self):
        pass

    def act(self, last_offer=None, last_demand=None):
        if last_offer is None:
            last_offer = np.random.rand(1)
        if last_demand is None:
            last_demand = np.random.rand(1)
        return np.array((last_offer, last_demand))


class BenchmarkBot:
    def __init__(self, benchmark=1):
        self.benchmark = benchmark
        self.cum_total = 0


class GreedFearBot:
    def __init__(self, greed=1, fear=1):
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
