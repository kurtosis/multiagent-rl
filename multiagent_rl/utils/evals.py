import numpy as np
import torch


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
