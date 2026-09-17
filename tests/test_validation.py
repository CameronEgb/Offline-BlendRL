import pytest
from unittest.mock import MagicMock, patch

from src.pipeline.validation import validate_experiment_config, PARADIGM_CONSTRAINTS
from src.pipeline.exceptions import ConfigurationError

@pytest.fixture
def mock_cfg():
    cfg = MagicMock()
    # Default values that pass for "online_v_offline"
    cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'online_v_offline',
        'task': 'rl',
        'online_methods': 'ppo/cp_tuned',
        'offline_methods': '',
        'intervals_count': 1,
        'eval_episodes': 100,
        'sweep': False,
        'group': '',
        'experiment_id': ''
    }.get(k, d)
    cfg.env.name = 'cartpole'
    cfg.env.offline_only = False
    cfg.env.get.side_effect = lambda k, d=None: {'offline_only': False}.get(k, d)
    return cfg

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_missing_paradigm(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': None,
        'task': 'rl'
    }.get(k, d)
    with pytest.raises(ConfigurationError, match="has no 'paradigm' declared"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_unknown_paradigm(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'unknown_paradigm',
        'task': 'rl'
    }.get(k, d)
    with pytest.raises(ConfigurationError, match="unknown paradigm 'unknown_paradigm'"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_offline_only_mismatch_requires_true(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'offline_only',
        'task': 'rl',
        'intervals_count': 1,
        'eval_episodes': 0,
        'online_methods': ''
    }.get(k, d)
    # env is offline_only = False, but paradigm requires True
    with pytest.raises(ConfigurationError, match="requires an offline-only environment"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_offline_only_mismatch_requires_false(mock_yaml, mock_cfg):
    # env is offline_only = True, but paradigm requires False
    mock_cfg.env.offline_only = True
    with pytest.raises(ConfigurationError, match="requires a live simulator environment"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_intervals_count_gt1_offline_only(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'offline_only',
        'task': 'rl',
        'intervals_count': 2,
        'eval_episodes': 0,
        'online_methods': ''
    }.get(k, d)
    mock_cfg.env.offline_only = True
    with pytest.raises(ConfigurationError, match="does not support intervals_count > 1"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_eval_episodes_offline_only(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'offline_only',
        'task': 'rl',
        'intervals_count': 1,
        'eval_episodes': 100,
        'online_methods': ''
    }.get(k, d)
    mock_cfg.env.offline_only = True
    with pytest.raises(ConfigurationError, match="disables simulated gym rollouts"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_missing_online_methods(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'online_v_offline',
        'task': 'rl',
        'intervals_count': 1,
        'eval_episodes': 100,
        'online_methods': ''
    }.get(k, d)
    with pytest.raises(ConfigurationError, match="requires at least one entry in 'online_methods'"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_successful_validation_returns_notices(mock_yaml, mock_cfg):
    # Valid online_v_offline setup
    notices = validate_experiment_config(mock_cfg, 'test_exp')
    assert isinstance(notices, list)

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_task_paradigms_bypass(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'task': 'early_prediction',
        'online_methods': '',
        'offline_methods': ''
    }.get(k, d)
    # Should bypass paradigm validation entirely and not raise any errors
    notices = validate_experiment_config(mock_cfg, 'test_exp')
    assert isinstance(notices, list)

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={'intervals_count': 5})
def test_explicit_intervals_count(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'offline_only',
        'task': 'rl',
        'eval_episodes': 0,
        'online_methods': ''
    }.get(k, d)
    mock_cfg.env.offline_only = True
    with pytest.raises(ConfigurationError, match="does not support intervals_count > 1"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={'eval_episodes': 5})
def test_explicit_eval_episodes(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'offline_only',
        'task': 'rl',
        'intervals_count': 1,
        'online_methods': ''
    }.get(k, d)
    mock_cfg.env.offline_only = True
    with pytest.raises(ConfigurationError, match="disables simulated gym rollouts"):
        validate_experiment_config(mock_cfg, 'test_exp')

@patch('src.pipeline.validation._load_raw_experiment_yaml', return_value={})
def test_sweep_maximize_offline_only(mock_yaml, mock_cfg):
    mock_cfg.get.side_effect = lambda k, d=None: {
        'paradigm': 'offline_only',
        'task': 'rl',
        'intervals_count': 1,
        'eval_episodes': 0,
        'online_methods': '',
        'hydra': {'sweeper': {'direction': 'maximize'}}
    }.get(k, d)
    mock_cfg.env.offline_only = True
    notices = validate_experiment_config(mock_cfg, 'test_exp', is_sweep=True)
    assert any("Optuna direction is 'maximize'" in n for n in notices)
