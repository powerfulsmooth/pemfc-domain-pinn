"""Independent manufactured scalar diffusion on eleven equal x-slabs of [-1,1]^3.

Only slot 4 has a PDE: -lap(T) = 3*pi^2*T_exact, T_exact=prod(sin(pi*x_j)).
Diffusivity is one, external temperature is zero, temperature and normal derivative
match at internal slab planes. Other active slots are supervised analytic examples.
This domain and its equations are NOT the original fuel cell.
"""
import math
import torch
from .model import ACTIVE
from .operators import gradient, laplacian
from .training import LossChunk


def exact_temperature(xyz):
    return torch.sin(math.pi * xyz).prod(dim=1)


def slab_tags(xyz):
    return torch.floor((xyz[:, 0] + 1) * 11 / 2).long().clamp(0, 10)


def sample_plan(n, device, generator):
    if n < 2:
        raise ValueError("Use at least two samples per slab")
    def uniform():
        return 2 * torch.rand((n, 3), generator=generator, dtype=torch.float64) - 1
    interior, interfaces = [], []
    for tag in range(11):
        x = uniform()
        x[:, 0] = -1 + (tag + (x[:, 0] + 1) / 2) * (2 / 11)
        interior.append(x.to(device))
    for left in range(10):
        x = uniform()
        x[:, 0] = -1 + (left + 1) * (2 / 11)
        interfaces.append(x.to(device))
    # Six equal-area faces of the manufactured cube.
    boundary = []
    for axis in range(3):
        for side in (-1., 1.):
            x = uniform()
            x[:, axis] = side
            boundary.append(x)
    return {"interior": interior, "interfaces": interfaces,
            "boundary": torch.cat(boundary).to(device)}


def loss_chunks(model, plan, physics):
    amplitudes = torch.linspace(.1, 1., 10, dtype=torch.float64,
                                device=model.onehot.device)
    amplitudes[4] = 1
    for tag, points in enumerate(plan["interior"]):
        x = points.detach().clone().requires_grad_(physics)
        prediction = model(x, tag)
        target = exact_temperature(x).detach()[:, None] * amplitudes
        active = list(ACTIVE[tag])
        yield LossChunk("data", (prediction[:, active] - target[:, active]).square().mean(), 1 / 11)
        if physics:
            # Rebuild after the data backward; do not reuse a freed graph.
            x = points.detach().clone().requires_grad_(True)
            temperature = model(x, tag)[:, 4]
            forcing = 3 * math.pi**2 * exact_temperature(x).detach()
            residual = (-laplacian(temperature, x) - forcing) / (3 * math.pi**2)
            yield LossChunk("PDE", residual.square().mean(), 1 / 11)
    if physics:
        x = plan["boundary"]
        yield LossChunk("BC", model(x, slab_tags(x))[:, 4].square().mean())
        for left, points in enumerate(plan["interfaces"]):
            x = points.detach().clone().requires_grad_(True)
            ql = model(x, left)[:, 4]
            qr = model(x, left + 1)[:, 4]
            value_jump = ql - qr
            flux_jump = (gradient(ql, x)[:, 0] - gradient(qr, x)[:, 0]) / math.pi
            yield LossChunk("interface", .5 * (value_jump.square() + flux_jump.square()).mean(), .1)
