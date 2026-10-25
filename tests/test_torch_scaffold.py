"""Run these tests in Colab or an environment with Torch; not run during packaging."""
import copy
import math
import pytest
import torch
from pemfc_public.model import DomainNetwork
from pemfc_public.operators import gradient, laplacian
from pemfc_public.training import accumulate, take_step, LossChunk
from pemfc_public.synthetic import exact_temperature, sample_plan, loss_chunks


def test_topology_routing_mask_and_point_independence():
    torch.manual_seed(31)
    model = DomainNetwork()
    assert sum(p.numel() for p in model.parameters()) == 13247
    xyz = torch.rand((22, 3), dtype=torch.float64, requires_grad=True)
    tags = torch.arange(11).repeat_interleave(2)
    mixed = model(xyz, tags)
    for tag in range(11):
        selection = tags == tag
        assert torch.allclose(mixed[selection], model(xyz[selection], tag), atol=1e-14)
    assert torch.equal(mixed * (1 - model.active_mask[tags]), torch.zeros_like(mixed))
    one = model(xyz[:1], 0)[:, 4]
    grad = torch.autograd.grad(one.sum(), xyz)[0]
    assert torch.equal(grad[1:], torch.zeros_like(grad[1:]))


def test_manufactured_3d_operators():
    x = torch.tensor([[.17, -.31, .26], [.37, .12, -.21]],
                     dtype=torch.float64, requires_grad=True)
    q = exact_temperature(x)
    assert torch.allclose(laplacian(q, x), -3 * math.pi**2 * q, atol=1e-12, rtol=1e-12)
    polynomial = x[:, 0]**3 + x[:, 1]**2 + x[:, 2]
    assert torch.allclose(gradient(polynomial, x),
                          torch.stack((3*x[:, 0]**2, 2*x[:, 1], torch.ones(2)), 1))


def test_chunked_gradients_equal_full_reduction():
    torch.manual_seed(7)
    first = DomainNetwork()
    second = copy.deepcopy(first)
    x = torch.rand((9, 3), dtype=torch.float64)
    target = torch.sin(x[:, 0])
    full = (first(x, 4)[:, 4] - target).square().mean()
    full.backward()
    def chunks():
        for sl in (slice(0, 2), slice(2, 6), slice(6, 9)):
            yield LossChunk("data", (second(x[sl], 4)[:, 4] - target[sl]).square().mean(),
                            len(x[sl]) / len(x))
    accumulate(second, chunks(), 0)
    for a, b in zip(first.parameters(), second.parameters()):
        assert (a.grad is None) == (b.grad is None)
        if a.grad is not None:
            assert torch.allclose(a.grad, b.grad, atol=1e-13, rtol=1e-12)


def test_real_synthetic_loss_graph_and_rollback():
    torch.manual_seed(9)
    model = DomainNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    plan = sample_plan(2, "cpu", torch.Generator().manual_seed(1))
    record = take_step(model, optimizer, loss_chunks(model, plan, True), .01)
    assert all(math.isfinite(v) for k, v in record.items() if k != "clipped")
    before = copy.deepcopy(model.state_dict())
    moments = copy.deepcopy(optimizer.state_dict())
    plan = sample_plan(2, "cpu", torch.Generator().manual_seed(2))
    with pytest.raises(FloatingPointError):
        take_step(model, optimizer, loss_chunks(model, plan, True), .01,
                  candidate_check=lambda: False)
    assert all(torch.equal(before[key], value) for key, value in model.state_dict().items())
    after = optimizer.state_dict()
    for key, state in moments["state"].items():
        for field, value in state.items():
            assert torch.equal(value, after["state"][key][field])
