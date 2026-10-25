"""Autograd for independent coordinate samples, not spatial-attention networks."""
import torch


def gradient(scalar, xyz):
    if scalar.ndim != 1 or not xyz.requires_grad:
        raise ValueError("Expected scalar (N,) and differentiable coordinates")
    if not scalar.requires_grad:
        raise ValueError("Detached scalar; spatial graph is required")
    value = torch.autograd.grad(scalar.sum(), xyz, create_graph=True,
                                retain_graph=True, allow_unused=True)[0]
    if value is None:
        raise ValueError("Scalar is disconnected from coordinates")
    return value


def divergence(vector, xyz):
    if vector.shape != xyz.shape:
        raise ValueError("Expected vector (N,3)")
    parts = []
    for axis in range(3):
        component = vector[:, axis]
        if component.requires_grad:
            deriv = torch.autograd.grad(component.sum(), xyz, create_graph=True,
                                        retain_graph=True, allow_unused=True)[0]
            parts.append(xyz[:, axis] * 0 if deriv is None else deriv[:, axis])
        else:
            parts.append(xyz[:, axis] * 0)
    return sum(parts)


def laplacian(scalar, xyz):
    return divergence(gradient(scalar, xyz), xyz)
