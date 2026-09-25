import torch
import torch.nn as nn

class SAMOptimizer:
    def __init__(self, params, base_optimizer, rho=0.05):
        self.params = list(params)
        self.base_optimizer = base_optimizer
        self.rho = rho
        self.state = {}

    @torch.no_grad()
    def first_step(self):
        norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for p in self.params if p.grad is not None]),
            p=2
        )
        scale = self.rho / (norm + 1e-12)
        for p in self.params:
            if p.grad is not None:
                self.state[p] = p.data.clone()
                e_w = p.grad * scale
                p.add_(e_w)

    @torch.no_grad()
    def second_step(self):
        for p in self.params:
            if p.grad is not None:
                p.data = self.state[p]
        self.base_optimizer.step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()
