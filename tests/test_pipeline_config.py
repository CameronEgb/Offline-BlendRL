"""Tests for pipeline configuration utilities (src/pipeline/config.py)."""

import pytest

from src.app.pipeline.config import normalize_agent_name


class TestNormalizeAgentName:
    """Tests for agent name normalization (slash to underscore, etc)."""

    def test_slash_to_underscore(self):
        result = normalize_agent_name("ppo/cp_tuned")
        assert "/" not in result
        assert "_" in result

    def test_already_normalized(self):
        result = normalize_agent_name("ppo_cp_tuned")
        assert result == "ppo_cp_tuned"

    def test_empty_string(self):
        result = normalize_agent_name("")
        assert result == ""
