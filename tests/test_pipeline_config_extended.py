import pytest
from src.pipeline.config import parse_method_list, resolve_experiment_config_name
import os

def test_parse_method_list_empty():
    assert parse_method_list(None) == []
    assert parse_method_list("") == []
    assert parse_method_list([]) == []

def test_parse_method_list_string():
    assert parse_method_list("ppo") == ["ppo"]

def test_parse_method_list_comma_separated():
    assert parse_method_list("ppo, cql, dqn") == ["ppo", "cql", "dqn"]
    assert parse_method_list("ppo,cql,dqn") == ["ppo", "cql", "dqn"]

def test_parse_method_list_list_input():
    assert parse_method_list(["ppo", "cql"]) == ["ppo", "cql"]

def test_parse_method_list_omegaconf_list():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(["ppo", "cql"])
    assert parse_method_list(cfg) == ["ppo", "cql"]

def test_resolve_experiment_config_name_direct_match(monkeypatch):
    class MockPath:
        def __init__(self, path_str):
            self.path_str = str(path_str)
        def exists(self):
            return True
        def __truediv__(self, other):
            return MockPath(f"{self.path_str}/{other}")
            
    monkeypatch.setattr("pathlib.Path", MockPath)
    # direct path exists will return True, so it will return clean_input
    assert resolve_experiment_config_name("my_exp") == "my_exp"
    assert resolve_experiment_config_name("my_exp.yaml") == "my_exp"

def test_resolve_experiment_config_name_exp_dir_missing(monkeypatch):
    class MockPathMissing:
        def __init__(self, path_str):
            self.path_str = str(path_str)
        def exists(self):
            return False
            
    monkeypatch.setattr("pathlib.Path", MockPathMissing)
    assert resolve_experiment_config_name("my_exp") == "my_exp"

def test_resolve_experiment_config_name_glob_single(monkeypatch):
    class MockGlobMatch:
        def relative_to(self, base):
            return MockGlobMatchSuffix(f"group/my_exp.yaml")
    class MockGlobMatchSuffix:
        def __init__(self, s): self.s = s
        def with_suffix(self, sfx): return self.s.replace(".yaml", sfx)
        
    class MockPathGlob:
        def __init__(self, path_str):
            self.path_str = str(path_str)
        def exists(self):
            # exp_dir exists, direct path doesn't
            if self.path_str == "in/config/experiment": return True
            return False
        def __truediv__(self, other):
            return MockPathGlob(f"{self.path_str}/{other}")
        def glob(self, pattern):
            return [MockGlobMatch()]
            
    monkeypatch.setattr("pathlib.Path", MockPathGlob)
    assert resolve_experiment_config_name("my_exp") == "group/my_exp"

def test_resolve_experiment_config_name_glob_multiple(monkeypatch):
    class MockPathGlobMult:
        def __init__(self, path_str):
            self.path_str = str(path_str)
        def exists(self):
            if self.path_str == "in/config/experiment": return True
            return False
        def __truediv__(self, other):
            return MockPathGlobMult(f"{self.path_str}/{other}")
        def glob(self, pattern):
            return ["match1", "match2"]
            
    monkeypatch.setattr("pathlib.Path", MockPathGlobMult)
    with pytest.raises(ValueError, match="Ambiguous experiment name"):
        resolve_experiment_config_name("my_exp")

def test_resolve_experiment_config_name_not_found(monkeypatch):
    class MockPathGlobNone:
        def __init__(self, path_str):
            self.path_str = str(path_str)
        def exists(self):
            if self.path_str == "in/config/experiment": return True
            return False
        def __truediv__(self, other):
            return MockPathGlobNone(f"{self.path_str}/{other}")
        def glob(self, pattern):
            return []
            
    monkeypatch.setattr("pathlib.Path", MockPathGlobNone)
    with pytest.raises(ValueError, match="not found in"):
        resolve_experiment_config_name("my_exp")
