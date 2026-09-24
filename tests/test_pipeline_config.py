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

    def test_hierarchical_agent_and_model_overrides(self):
        """params.agent.<algo> and params.model.<arch> should apply selectively to matching methods."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "epochs_per_interval": 25,
                    "eval_interval_epochs": 2,
                    "gamma": 0.95,
                    "agent": {
                        "cql": {
                            "cql_alpha": 0.1,
                            "batch_size": 1024,
                            "lr": 3e-4,
                        },
                        "ppo": {
                            "clip_coef": 0.2,
                            "batch_size": 64,
                        },
                    },
                    "model": {
                        "dnn": {
                            "hidden_sizes": [512, 512],
                        },
                        "blendrl": {
                            "blend_q_values": True,
                            "rules": "rigid",
                        },
                    },
                },
                "cql_dnn": {
                    "agent": "cql",
                    "model": "dnn",
                },
                "cql_dueling_resnet": {
                    "agent": "cql",
                    "model": "dueling_resnet",
                },
                "ppo_dnn": {
                    "agent": "ppo",
                    "model": "dnn",
                },
            }
        }
        res = parse_methods_dict(cfg)
        assert len(res) == 3

        # cql_dnn gets universal globals + agent.cql + model.dnn
        assert res["cql_dnn"]["agent"] == "cql"
        assert res["cql_dnn"]["model"] == "dnn"
        assert res["cql_dnn"]["epochs_per_interval"] == 25
        assert res["cql_dnn"]["gamma"] == 0.95
        assert res["cql_dnn"]["cql_alpha"] == 0.1
        assert res["cql_dnn"]["batch_size"] == 1024
        assert res["cql_dnn"]["lr"] == 3e-4
        assert res["cql_dnn"]["hidden_sizes"] == [512, 512]
        assert "clip_coef" not in res["cql_dnn"]
        assert "rules" not in res["cql_dnn"]

        # cql_dueling_resnet gets agent.cql but NOT model.dnn (relies on dueling_resnet defaults)
        assert res["cql_dueling_resnet"]["agent"] == "cql"
        assert res["cql_dueling_resnet"]["model"] == "dueling_resnet"
        assert res["cql_dueling_resnet"]["cql_alpha"] == 0.1
        assert res["cql_dueling_resnet"]["batch_size"] == 1024
        assert "hidden_sizes" not in res["cql_dueling_resnet"]

        # ppo_dnn gets agent.ppo + model.dnn, but NOT agent.cql
        assert res["ppo_dnn"]["agent"] == "ppo"
        assert res["ppo_dnn"]["model"] == "dnn"
        assert res["ppo_dnn"]["clip_coef"] == 0.2
        assert res["ppo_dnn"]["batch_size"] == 64
        assert res["ppo_dnn"]["hidden_sizes"] == [512, 512]
        assert "cql_alpha" not in res["ppo_dnn"]

    def test_method_override_precedence_over_universal(self):
        """Method-level declarations must override both universal globals and agent/model blocks."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "agent": {
                        "cql": {
                            "lr": 3e-4,
                            "batch_size": 1024,
                        }
                    },
                    "model": {
                        "dnn": {
                            "hidden_sizes": [512, 512],
                        }
                    },
                },
                "cql_custom": {
                    "agent": "cql",
                    "model": "dnn",
                    "batch_size": 256,
                    "hidden_sizes": [1024, 512, 256],
                },
            }
        }
        res = parse_methods_dict(cfg)
        assert res["cql_custom"]["lr"] == 3e-4
        assert res["cql_custom"]["batch_size"] == 256
        assert res["cql_custom"]["hidden_sizes"] == [1024, 512, 256]

    def test_blendrl_inherits_neural_and_symbolic_defaults(self):
        """BlendRL without submodel overrides must inherit default dnn and nsfr configs."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "cql_blendrl": {
                    "agent": "cql",
                    "model": "blendrl",
                }
            }
        }
        res = parse_methods_dict(cfg)
        m = res["cql_blendrl"]
        assert m["model"] == "blendrl"
        assert m["model_params"]["neural"]["architecture"] == "dnn"
        assert m["model_params"]["neural"]["hidden_sizes"] == [256, 256]
        assert m["model_params"]["symbolic"]["type"] == "nsfr"
        assert m["model_params"]["symbolic"]["rules"] == "default"
        assert m["model_params"]["blender"]["mode"] == "neural"
        assert m["model_params"]["blender"]["blend_function"] == "softmax"

    def test_blendrl_neural_overwrite_inherits_base_config(self):
        """Specifying neural: dueling_resnet must automatically inherit its hidden_sizes from dueling_resnet.yaml."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "cql_blendrl_resnet": {
                    "agent": "cql",
                    "model": "blendrl",
                    "neural": "dueling_resnet",
                    "symbolic": {"rules": "rigid"},
                }
            }
        }
        res = parse_methods_dict(cfg)
        m = res["cql_blendrl_resnet"]
        # Neural actor inherited dueling_resnet.yaml defaults!
        assert m["model_params"]["neural"]["architecture"] == "dueling_resnet"
        assert m["model_params"]["neural"]["hidden_sizes"] == [512, 512, 256, 128]
        # Symbolic actor inherited nsfr.yaml defaults with rules: rigid
        assert m["model_params"]["symbolic"]["type"] == "nsfr"
        assert m["model_params"]["symbolic"]["rules"] == "rigid"

    def test_blendrl_symbolic_cew_inherits_cew_defaults(self):
        """Specifying symbolic: { name: cew, ecm_dthr: 0.03 } must inherit fyd and fyd_top_k from cew.yaml."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "cql_blendrl_cew": {
                    "agent": "cql",
                    "model": "blendrl",
                    "neural": "dueling_resnet",
                    "symbolic": {
                        "name": "cew",
                        "ecm_dthr": 0.03,
                    },
                }
            }
        }
        res = parse_methods_dict(cfg)
        m = res["cql_blendrl_cew"]
        sym = m["model_params"]["symbolic"]
        assert sym["type"] == "cew"
        assert sym["ecm_dthr"] == 0.03
        assert sym["fyd"] is False
        assert sym["fyd_top_k"] == 50

    def test_blendrl_universal_overrides_in_params(self):
        """Universal params.model.blendrl overrides should apply to all BlendRL methods."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "params": {
                    "model": {
                        "blendrl": {
                            "blend_q_values": True,
                            "neural": "dueling_resnet",
                        }
                    }
                },
                "method_a": {
                    "agent": "cql",
                    "model": "blendrl",
                    "symbolic": {"rules": "rigid"},
                },
                "method_b": {
                    "agent": "cql",
                    "model": "blendrl",
                    "symbolic": {"name": "cew", "ecm_dthr": 0.05},
                },
            }
        }
        res = parse_methods_dict(cfg)
        for m_name in ("method_a", "method_b"):
            m = res[m_name]
            assert m["model_params"]["blend_q_values"] is True
            assert m["model_params"]["neural"]["architecture"] == "dueling_resnet"
            assert m["model_params"]["neural"]["hidden_sizes"] == [512, 512, 256, 128]

        assert res["method_a"]["model_params"]["symbolic"]["rules"] == "rigid"
        assert res["method_b"]["model_params"]["symbolic"]["type"] == "cew"
        assert res["method_b"]["model_params"]["symbolic"]["ecm_dthr"] == 0.05

    def test_blendrl_fully_hierarchical_nesting(self):
        """Hierarchical specification: model.blendrl.neural and model.blendrl.symbolic.cew."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "cql_blendrl_hierarchical": {
                    "agent": "cql",
                    "model": {
                        "blendrl": {
                            "neural": "dueling_resnet",
                            "symbolic": {
                                "cew": {
                                    "ecm_dthr": 0.03
                                }
                            }
                        }
                    }
                }
            }
        }
        res = parse_methods_dict(cfg)
        m = res["cql_blendrl_hierarchical"]
        assert m["model"]["name"] == "blendrl"
        assert m["model_params"]["neural"]["architecture"] == "dueling_resnet"
        assert m["model_params"]["neural"]["hidden_sizes"] == [512, 512, 256, 128]
        assert m["model_params"]["symbolic"]["type"] == "cew"
        assert m["model_params"]["symbolic"]["ecm_dthr"] == 0.03
        assert m["model_params"]["symbolic"]["fyd"] is False
        assert m["model_params"]["symbolic"]["fyd_top_k"] == 50

    def test_blendrl_fully_hierarchical_neural_and_symbolic_params(self):
        """Hierarchical specification: neural.dueling_resnet and symbolic.nsfr."""
        from src.app.pipeline.config import parse_methods_dict

        cfg = {
            "methods": {
                "cql_blendrl_custom": {
                    "agent": "cql",
                    "model": {
                        "blendrl": {
                            "neural": {
                                "dueling_resnet": {
                                    "hidden_sizes": [256, 128]
                                }
                            },
                            "symbolic": {
                                "nsfr": {
                                    "rules": "rigid"
                                }
                            }
                        }
                    }
                }
            }
        }
        res = parse_methods_dict(cfg)
        m = res["cql_blendrl_custom"]
        assert m["model_params"]["neural"]["architecture"] == "dueling_resnet"
        assert m["model_params"]["neural"]["hidden_sizes"] == [256, 128]
        assert m["model_params"]["symbolic"]["type"] == "nsfr"
        assert m["model_params"]["symbolic"]["rules"] == "rigid"



