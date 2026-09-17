"""Core data types and contracts for NeSyRL."""
from typing import Any

import torch


class ActionResult:
    """Standard polymorphic return container for get_action_and_value().

    Supports:
      - Attribute access: res.action, res.logprob, res.entropy, res.value, res.aux
      - Backward-compatible 4-element unpacking: action, logprob, entropy, value = res
      - Backward-compatible indexing: res[0]
      - Auxiliary property access: res.blend_entropy
    """

    def __init__(
        self,
        action: torch.Tensor,
        logprob: torch.Tensor,
        entropy: torch.Tensor,
        value: torch.Tensor,
        aux: dict[str, Any] | None = None,
    ):
        self.action = action
        self.logprob = logprob
        self.entropy = entropy
        self.value = value
        self.aux: dict[str, Any] = aux if aux is not None else {}

    def __iter__(self):
        """Yield the 4 standard Actor-Critic elements for backward-compatible unpacking."""
        yield self.action
        yield self.logprob
        yield self.entropy
        yield self.value

    def __getitem__(self, index: int):
        return (self.action, self.logprob, self.entropy, self.value)[index]

    def __len__(self):
        return 4

    @property
    def blend_entropy(self) -> torch.Tensor:
        """Return blend_entropy if present in aux, else scalar 0.0 on same device as action."""
        if "blend_entropy" in self.aux:
            return self.aux["blend_entropy"]
        device = getattr(self.action, "device", None)
        return torch.tensor(0.0, device=device)

    def __repr__(self):
        return (
            f"ActionResult(action={self.action.shape if hasattr(self.action, 'shape') else self.action}, "
            f"value={self.value.shape if hasattr(self.value, 'shape') else self.value}, "
            f"aux_keys={list(self.aux.keys())})"
        )
