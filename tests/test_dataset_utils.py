import pytest
import numpy as np
import torch
import pickle
import os
from src.dataset_utils import DatasetWriter

@pytest.fixture
def writer(tmp_path):
    return DatasetWriter(save_dir=tmp_path / "dataset", chunk_size=10, env_name="test_env")

def test_writer_init_creates_dir(tmp_path):
    DatasetWriter(save_dir=tmp_path / "new_dataset", chunk_size=10)
    assert (tmp_path / "new_dataset").exists()

def test_writer_add_5_param(writer):
    obs = torch.zeros((4,))
    action = torch.tensor(1)
    reward = torch.tensor(1.0)
    next_obs = torch.ones((4,))
    done = torch.tensor(False)

    writer.add(obs, action, reward, next_obs, done, logic_obs=np.array([1, 0]), next_logic_obs=np.array([0, 1]))
    assert len(writer.buffer) == 1
    t = writer.buffer[0]
    assert "obs" in t
    assert "logic_obs" in t
    assert np.array_equal(t["logic_obs"], np.array([1, 0]))

def test_writer_add_7_param(writer):
    obs = np.zeros((4,))
    logic_obs = np.zeros((2,))
    action = 1
    reward = 1.0
    next_obs = np.ones((4,))
    next_logic_obs = np.ones((2,))
    done = False

    writer.add(obs, logic_obs, action, reward, next_obs, next_logic_obs, done)
    assert len(writer.buffer) == 1
    t = writer.buffer[0]
    assert np.array_equal(t["obs"], obs)
    assert np.array_equal(t["logic_obs"], logic_obs)

def test_writer_flush_writes_pickle(writer, tmp_path):
    obs = np.zeros((4,))
    action = 1
    reward = 1.0
    next_obs = np.ones((4,))
    done = False
    writer.add(obs, action, reward, next_obs, done)
    
    writer.flush()
    assert len(list((tmp_path / "dataset").glob("*.pkl"))) == 1

def test_writer_round_trip(writer, tmp_path):
    obs = np.array([1, 2, 3, 4], dtype=np.float32)
    action = 1
    reward = 1.0
    next_obs = np.array([5, 6, 7, 8], dtype=np.float32)
    done = True

    writer.add(obs, action, reward, next_obs, done)
    writer.flush()
    
    pkl_files = list((tmp_path / "dataset").glob("*.pkl"))
    assert len(pkl_files) == 1
    
    with open(pkl_files[0], "rb") as f:
        data = pickle.load(f)
        
    assert len(data) == 1
    t = data[0]
    assert np.array_equal(t["obs"], obs)
    assert t["action"] == action
    assert t["reward"] == reward
    assert np.array_equal(t["next_obs"], next_obs)
    assert t["done"] == done

def test_writer_empty_flush(writer, tmp_path):
    writer.flush()
    assert len(list((tmp_path / "dataset").glob("*.pkl"))) == 0
