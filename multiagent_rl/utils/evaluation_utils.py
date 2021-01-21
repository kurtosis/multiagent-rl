from copy import deepcopy
from itertools import chain
import os

import numpy as np
import pandas as pd

import torch
from torch.optim import Adam

def eval_q_vs_a(data, ac):
    o, a = data["obs"], data["act"]
    print("Varying offer")
    for i in np.arange(0, 1.1, 0.1):
        a[:, 0] = i
        q = ac.q(o, a)
        print(np.round(i, 2), np.round(q.mean().item(), 4))
    print("---")
    for i in np.arange(0.45, 0.56, 0.01):
        a[:, 0] = i
        q = ac.q(o, a)
        print(np.round(i, 2), np.round(q.mean().item(), 4))
    print("---")
    a = data["act"]
    print("Varying demand")
    for i in np.arange(0, 1.1, 0.1):
        a[:, 1] = i
        q = ac.q(o, a)
        print(np.round(i, 2), np.round(q.mean().item(), 4))
    print("---")
    for i in np.arange(0.45, 0.56, 0.01):
        a[:, 1] = i
        q = ac.q(o, a)
        print(np.round(i, 2), np.round(q.mean().item(), 4))
    print("---")


def print_vals(i, o, a, ac, col=0):
    a[:, col] = i
    # q1 = ac.q1(o, a)
    # q2 = ac.q2(o, a)
    q1 = ac.q1(torch.cat([o, a], dim=-1))
    q2 = ac.q2(torch.cat([o, a], dim=-1))
    print(np.round(i, 2), np.round(q1.mean().item(), 4), np.round(q2.mean().item(), 4))


def eval_q_td3(data, ac):
    o = data["obs"]
    a = deepcopy(data["act"])
    print("Varying offer")
    for i in np.arange(0, 1.1, 0.1):
        print_vals(i, o, a, ac)
    print("---")
    a = deepcopy(data["act"])
    for i in np.arange(0.45, 0.56, 0.01):
        print_vals(i, o, a, ac)
    print("---")
    a = deepcopy(data["act"])
    print("Varying demand")
    for i in np.arange(0, 1.1, 0.1):
        print_vals(i, o, a, ac, col=1)
    print("---")
    a = deepcopy(data["act"])
    for i in np.arange(0.45, 0.56, 0.01):
        print_vals(i, o, a, ac, col=1)
    print("---")


def eval_q_vs_a_2(data, ac):
    o, a = data["obs"], data["act"]

    a[:, 0] = 0.0
    a[:, 1] = 1.0
    q = ac.q(torch.cat([o, a], dim=-1))
    print(np.round(q.mean().item(), 4))

    a[:, 0] = 0.01
    a[:, 1] = 1.0
    q = ac.q(torch.cat([o, a], dim=-1))
    print(np.round(q.mean().item(), 4))

    a[:, 0] = 0.0
    a[:, 1] = 0.99
    q = ac.q(torch.cat([o, a], dim=-1))
    print(np.round(q.mean().item(), 4))

    print("Varying offer")
    for i in np.arange(0, 1.1, 0.1):
        a[:, 0] = i
        q = ac.q(torch.cat([o, a], dim=-1))
        print(np.round(i, 2), np.round(q.mean().item(), 4))
    print("---")
    for i in np.arange(0.45, 0.56, 0.01):
        a[:, 0] = i
        q = ac.q(torch.cat([o, a], dim=-1))
        print(np.round(i, 2), np.round(q.mean().item(), 4))
    print("---")
    a = data["act"]
    print("Varying demand")
    for i in np.arange(0, 1.1, 0.1):
        a[:, 1] = i
        q = ac.q(torch.cat([o, a], dim=-1))
        print(np.round(i, 2), np.round(q.mean().item(), 4))
    print("---")
    for i in np.arange(0.45, 0.56, 0.01):
        a[:, 1] = i
        q = ac.q(torch.cat([o, a], dim=-1))
        print(np.round(i, 2), np.round(q.mean().item(), 4))
    print("---")


def eval_a(data, ac):
    o = data["obs"]
    a = ac.pi(o)
    print("---")
    print(f"mean: {a.mean(dim=0)}")
    print(f"std: {a.std(dim=0)}")
    print(f"max: {a.max(dim=0).values}")
    print(f"min: {a.min(dim=0).values}")
    print("---")


def manual_reward(a, opp_offer, opp_demand):
    succeeded = (a[:, 0] >= opp_demand) * (a[:, 1] <= opp_offer)
    value = (1 - a[:, 0]) + opp_offer
    reward = succeeded * value
    return reward


def save_q_map(q, filename, step, n=101):
    """Evaluate a Q function on a grid over the action space, for a given observation. Write the results to file."""
    grid = np.linspace(0, 1, n)
    a0, a1 = np.meshgrid(grid, grid)
    a0 = a0.reshape(-1)
    a1 = a1.reshape(-1)
    a = torch.tensor(np.stack((a0, a1), axis=1), dtype=torch.float32)
    o = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
    o = o.repeat(a.shape[0], 1)
    with torch.no_grad():
        q_estim = q(torch.cat([o, a], dim=-1))
        # q_estim = q(o, a)
    result = torch.cat([o, a, q_estim.unsqueeze(dim=1)], dim=1)
    cols = (
        [f"o_{i}" for i in range(o.shape[-1])]
        + [f"a_{i}" for i in range(a.shape[-1])]
        + ["q"]
    )
    result = pd.DataFrame(result.numpy(), columns=cols)
    result["step"] = step
    if not os.path.isfile(os.path.expanduser(filename)):
        result.to_csv(filename, index=False, header="column_names")
    else:
        result.to_csv(filename, index=False, mode="a", header=False)


def check_q_backup():
    condn = (a[:, 0] > 0.9) * (a[:, 1] < 0.3)
    ff = torch.cat((a, backup.unsqueeze(dim=1)), dim=1)
    ff = torch.cat((ff, q1.unsqueeze(dim=1)), dim=1)
    ff = ff[condn]
    print(ff)

def eval_done(agent, agent_target, data, gamma, alpha, q_lr, n=101):
        o, a, r, obs_next, d = (
            data["obs"],
            data["act"],
            data["rew"],
            data["obs2"],
            data["done"],
        )
        agent_a = deepcopy(agent)

        qa_params = chain(agent_a.q1.parameters(), agent_a.q2.parameters())
        qa_optimizer = Adam(qa_params, lr=q_lr)

        # d = torch.zeros_like(d)

        q1 = agent_a.q1(torch.cat([o, a], dim=-1))
        q2 = agent_a.q2(torch.cat([o, a], dim=-1))

        # Bellman backup for Q function
        with torch.no_grad():
            a_next, logprob_next = agent_a.pi(obs_next)

            q1_target = agent_target.q1(torch.cat([obs_next, a_next], dim=-1))
            q2_target = agent_target.q2(torch.cat([obs_next, a_next], dim=-1))
            q_target = torch.min(q1_target, q2_target)
            backup = r + gamma * (1 - d) * (q_target - alpha * logprob_next)

        # MSE loss against Bellman backup
        loss_qa = ((q1 - backup) ** 2).mean() + ((q2 - backup) ** 2).mean()

        qa_optimizer.zero_grad()
        loss_qa.backward()
        qa_optimizer.step()

        grid = np.linspace(0, 1, n)
        a0, a1 = np.meshgrid(grid, grid)
        a0 = a0.reshape(-1)
        a1 = a1.reshape(-1)
        a = torch.tensor(np.stack((a0, a1), axis=1), dtype=torch.float32)
        o = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
        o = o.repeat(a.shape[0], 1)