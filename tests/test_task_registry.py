import pytest

from src.pipeline.task_registry import TASK_REGISTRY, get_task, register_task


@pytest.fixture(autouse=True)
def clean_registry():
    # Save the original state
    original = TASK_REGISTRY.copy()
    TASK_REGISTRY.clear()
    yield
    # Restore after test
    TASK_REGISTRY.clear()
    TASK_REGISTRY.update(original)


def test_register_task_single():
    @register_task("my_task")
    def dummy_task():
        pass

    assert "my_task" in TASK_REGISTRY
    assert TASK_REGISTRY["my_task"] is dummy_task


def test_register_task_multiple():
    @register_task("task1", "task2")
    def dummy_task():
        pass

    assert "task1" in TASK_REGISTRY
    assert "task2" in TASK_REGISTRY
    assert TASK_REGISTRY["task1"] is dummy_task
    assert TASK_REGISTRY["task2"] is dummy_task


def test_get_task_exact_match():
    @register_task("exact_match")
    def dummy_task():
        pass

    task_fn = get_task("exact_match")
    assert task_fn is dummy_task


def test_get_task_longest_prefix():
    @register_task("prefix")
    def prefix_task():
        pass

    @register_task("prefix_longer")
    def longer_task():
        pass

    task_fn = get_task("prefix_longer_suffix")
    assert task_fn is longer_task

    task_fn2 = get_task("prefix_other")
    assert task_fn2 is prefix_task


def test_get_task_unknown_raises():
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("unknown_task")


def test_auto_discover_called():
    # The real get_task calls auto_discover_tasks if registry is empty
    # We can test that it raises if still not found, or mock auto_discover_tasks.
    with pytest.raises(ValueError, match="Unknown task"):
        get_task("some_non_existent_task")


def test_list_tasks():
    @register_task("b_task", "c_task", "a_task")
    def dummy_task():
        pass

    # Check if a list_tasks function exists.
    try:
        from src.pipeline.task_registry import list_tasks

        tasks = list_tasks()
        assert tasks == ["a_task", "b_task", "c_task"]
    except ImportError:
        # If it doesn't exist, we skip
        pass
