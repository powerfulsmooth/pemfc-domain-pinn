"""Deterministic expert topology; inputs/outputs here are dimensionless.

No private CFD normalization, initialization offsets or physical decoder is used.
Only the assigned expert is evaluated at a point. Spatial samples are independent.
"""
import torch
from torch import nn
from torch.nn import functional as F


SPECS = (
    ("MEA", (4, 5, 6), (6, 48, 48, 48, 10), tuple(range(10)), 0, 10),
    ("GC_GDL", (1, 2, 3, 7, 8, 9), (9, 48, 48, 48, 9),
     (0, 1, 2, 3, 4, 5, 6, 8, 9), 10, 19),
    ("BP", (0, 10), (5, 24, 24, 24, 2), (4, 8), 19, 21),
)
ACTIVE = (
    (4, 8),
    (0, 1, 2, 3, 4, 5, 6, 9),
    (0, 1, 2, 3, 4, 5, 6, 8, 9),
    (0, 1, 2, 3, 4, 5, 6, 8, 9),
    tuple(range(10)),
    (4, 6, 7, 9),
    tuple(range(10)),
    (0, 1, 2, 3, 4, 5, 6, 8, 9),
    (0, 1, 2, 3, 4, 5, 6, 8, 9),
    (0, 1, 2, 3, 4, 5, 6, 9),
    (4, 8),
)


class DenseTanh(nn.Module):
    def __init__(self, widths, zero_output=False):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(a, b, dtype=torch.float64)
                                   for a, b in zip(widths[:-1], widths[1:]))
        for layer in self.layers[:-1]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        if zero_output:
            nn.init.zeros_(self.layers[-1].weight)
        else:
            nn.init.normal_(self.layers[-1].weight, std=.001)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        for layer in self.layers[:-1]:
            features = torch.tanh(layer(features))
        return self.layers[-1](features)


class DomainNetwork(nn.Module):
    """Three experts plus residual head. Coordinates must already be normalized."""
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleDict({s[0]: DenseTanh(s[2]) for s in SPECS})
        self.readout = DenseTanh((32, 16, 10), zero_output=True)
        mask = torch.zeros((11, 10), dtype=torch.float64)
        for tag, slots in enumerate(ACTIVE):
            mask[tag, list(slots)] = 1
        self.register_buffer("active_mask", mask)
        self.register_buffer("onehot", torch.eye(11, dtype=torch.float64))
        for name, _, _, slots, _, _ in SPECS:
            projection = torch.zeros((len(slots), 10), dtype=torch.float64)
            projection[torch.arange(len(slots)), list(slots)] = 1
            self.register_buffer("projection_" + name, projection)

    def forward_tag(self, xyz, tag):
        if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype != torch.float64:
            raise ValueError("Expected float64 (N,3) dimensionless coordinates")
        if type(tag) is not int or not 0 <= tag < 11:
            raise ValueError("Expected Python integer tag 0..10")
        if xyz.device != self.onehot.device:
            raise ValueError("Coordinates and model must use the same device")
        name, tags, _, slots, start, stop = next(s for s in SPECS if tag in s[1])
        local = xyz.new_zeros((len(xyz), len(tags)))
        local[:, tags.index(tag)] = 1
        raw = self.experts[name](torch.cat((xyz, local), dim=1))
        raw = raw * self.active_mask[tag, list(slots)]
        base = raw @ getattr(self, "projection_" + name)
        packed = F.pad(raw, (start, 21 - stop))
        head_features = torch.cat((packed, self.onehot[tag].expand(len(xyz), -1)), dim=1)
        return (base + self.readout(head_features)) * self.active_mask[tag]

    def forward(self, xyz, tags):
        if type(tags) is int:
            return self.forward_tag(xyz, tags)
        if tags.ndim != 1 or len(tags) != len(xyz) or tags.dtype != torch.long:
            raise ValueError("Expected one int64 material tag per coordinate")
        if tags.device != xyz.device or bool(((tags < 0) | (tags > 10)).any()):
            raise ValueError("Invalid material tags/device")
        result = xyz.new_zeros((len(xyz), 10))
        for tag in range(11):
            indices = torch.where(tags == tag)[0]
            if len(indices):
                result = result.index_copy(0, indices, self.forward_tag(xyz[indices], tag))
        return result
