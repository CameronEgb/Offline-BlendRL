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


class TestParseMethodsDict:
    """Tests for parse_methods_dict and shared parameter inheritance."""

    def test_basic_methods_without_params(self):
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "cql_dnn": {"agent": "cql", "model": "dnn", "lr": 1e-3},
                "cql_resnet": {"agent": "cql", "model": "dueling_resnet"},
            }
        }
        res = parse_methods_dict(cfg)
        assert len(res) == 2
        assert res["cql_dnn"]["agent"] == "cql"
        assert res["cql_dnn"]["lr"] == 1e-3
        assert res["cql_resnet"]["model"] == "dueling_resnet"

    def test_shared_params_under_methods(self):
        """methods.params should be merged into all methods and excluded from returned keys."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "agent": "cql",
                    "epochs_per_interval": 25,
                    "lr": 3e-4,
                },
                "cql_dnn": {"model": "dnn"},
                "cql_resnet": {"model": "dueling_resnet", "lr": 1e-4},
            }
        }
        res = parse_methods_dict(cfg)
        assert "params" not in res
        assert len(res) == 2
        # cql_dnn inherited agent and lr
        assert res["cql_dnn"]["agent"] == "cql"
        assert res["cql_dnn"]["model"] == "dnn"
        assert res["cql_dnn"]["epochs_per_interval"] == 25
        assert res["cql_dnn"]["lr"] == 3e-4
        # cql_resnet overridden lr
        assert res["cql_resnet"]["agent"] == "cql"
        assert res["cql_resnet"]["model"] == "dueling_resnet"
        assert res["cql_resnet"]["epochs_per_interval"] == 25
        assert res["cql_resnet"]["lr"] == 1e-4

    def test_top_level_params(self):
        """Top-level params should be merged into all methods."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "params": {"epochs_per_interval": 50, "gamma": 0.95},
            "methods": {
                "ppo_dnn": {"agent": "ppo", "model": "dnn"},
            }
        }
        res = parse_methods_dict(cfg)
        assert res["ppo_dnn"]["epochs_per_interval"] == 50
        assert res["ppo_dnn"]["gamma"] == 0.95

    def test_nested_deep_merge(self):
        """Nested dictionaries should be deep-merged rather than completely replaced."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "agent": "cql",
                    "blender": {"mode": "neural", "blend_function": "softmax"},
                },
                "cql_blendrl": {
                    "model": "blendrl",
                    "blender": {"blend_function": "gumbel"},
                },
            }
        }
        res = parse_methods_dict(cfg)
        assert res["cql_blendrl"]["blender"]["mode"] == "neural"
        assert res["cql_blendrl"]["blender"]["blend_function"] == "gumbel"

    def test_reserved_keys_excluded(self):
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {"lr": 1e-3},
                "_params_": {"gamma": 0.99},
                "defaults": {"epochs": 10},
                "common": {"batch_size": 32},
                "real_method": {"agent": "ppo", "model": "dnn"},
            }
        }
        res = parse_methods_dict(cfg)
        assert list(res.keys()) == ["real_method"]
        assert res["real_method"]["lr"] == 1e-3
        assert res["real_method"]["gamma"] == 0.99
        assert res["real_method"]["epochs"] == 10
        assert res["real_method"]["batch_size"] == 32

