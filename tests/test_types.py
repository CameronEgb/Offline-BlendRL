"""Unit tests for core data types and ActionResult container."""

import torch

from src.core.types import ActionResult


class TestActionResult:
    """Verify ActionResult contract: attribute access, 4-tuple unpacking, and aux handling."""

    def test_attribute_access(self):
        act = torch.tensor([1, 0])
        lp = torch.tensor([-0.69, -1.2])
        ent = torch.tensor(0.5)
        val = torch.tensor([10.0])
        res = ActionResult(action=act, logprob=lp, entropy=ent, value=val, aux={"blend_entropy": torch.tensor(0.25)})

        assert torch.equal(res.action, act)
        assert torch.equal(res.logprob, lp)
        assert torch.equal(res.entropy, ent)
        assert torch.equal(res.value, val)
        assert res.blend_entropy == 0.25

    def test_four_tuple_unpacking(self):
        """Ensure backward compatibility with standard 4-element unpacking."""
        res = ActionResult(
            action=torch.tensor(0),
            logprob=torch.tensor(-0.1),
            entropy=torch.tensor(0.3),
            value=torch.tensor(5.0),
        )
        action, logprob, entropy, value = res
        assert action == 0
        assert logprob == -0.1
        assert entropy == 0.3
        assert value == 5.0

    def test_indexing(self):
        res = ActionResult(
            action=torch.tensor(2),
            logprob=torch.tensor(-0.5),
            entropy=torch.tensor(0.1),
            value=torch.tensor(1.0),
        )
        assert res[0] == 2
        assert res[3] == 1.0

    def test_default_blend_entropy(self):
        """When aux has no blend_entropy, it should gracefully return scalar 0.0."""
        res = ActionResult(
            action=torch.tensor(1),
            logprob=torch.tensor(-0.2),
            entropy=torch.tensor(0.4),
            value=torch.tensor(2.0),
        )
        assert res.blend_entropy == 0.0
