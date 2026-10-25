"""Generic loss accumulation. Exact private sampling/physics gates are withheld."""
import copy
from dataclasses import dataclass
import math
import torch


CATEGORIES = ("data", "PDE", "BC", "interface")


@dataclass
class LossChunk:
    category: str
    loss: torch.Tensor
    coefficient: float = 1.


def accumulate(model, chunks, alpha):
    """Backpropagate chunks at unchanged parameters; coefficients include reductions."""
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Expected finite alpha in [0,1]")
    model.zero_grad(set_to_none=True)
    record = {name: 0. for name in CATEGORIES}
    seen = False
    for item in chunks:
        if item.category not in record or item.loss.ndim != 0:
            raise ValueError("Expected a named scalar loss")
        if not math.isfinite(item.coefficient) or item.coefficient <= 0:
            raise ValueError("Expected a finite positive reduction coefficient")
        if not bool(torch.isfinite(item.loss)):
            raise FloatingPointError("Nonfinite loss")
        if alpha == 0 and item.category != "data":
            raise ValueError("Warmup must supply data-only chunks")
        value = item.loss * item.coefficient
        record[item.category] += float(value.detach())
        (value * (1 if item.category == "data" else alpha)).backward()
        seen = True
    if not seen:
        raise ValueError("Empty update")
    record["objective"] = record["data"] + alpha * sum(record[n] for n in CATEGORIES[1:])
    if not math.isfinite(record["objective"]):
        raise FloatingPointError("Nonfinite objective")
    return record


def take_step(model, optimizer, chunks, alpha, clip=1., candidate_check=None):
    """One logical Adam update; rollback model AND moments on nonfinite/rejected state.

    This is a numerical guard, not a loss-descent line search. The caller owns sample
    generation and RNG. Retrying must replay the same saved plan explicitly.
    """
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("Expected finite positive clipping threshold")
    old_model = copy.deepcopy(model.state_dict())
    old_optimizer = copy.deepcopy(optimizer.state_dict())
    try:
        record = accumulate(model, chunks, alpha)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
        if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
            raise FloatingPointError("Nonfinite candidate parameters")
        if candidate_check is not None and not candidate_check():
            raise FloatingPointError("Candidate rejected")
    except Exception:
        model.load_state_dict(old_model)
        optimizer.load_state_dict(old_optimizer)
        optimizer.zero_grad(set_to_none=True)
        raise
    record.update(gradient_norm_before_clip=float(norm), clipped=bool(norm > clip))
    return record
