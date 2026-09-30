"""Reads the Hydra config tree and answers experiment-compatibility questions.

Pure Python: no Qt and no torch imports, so it loads in headless CI and can be
unit tested without a display or the ML stack.

The panel asks this module three things: which options are valid for a given
paradigm, which form fields that paradigm permits, and what command line a
selection turns into.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

# Paradigms opt into fields with an `allows_<flag>` key whose name does not
# always match the Hydra key the field maps to.
_ALLOW_FLAGS = {
    "eval_episodes": "allows_eval_episodes",
    "intervals_count": "allows_intervals",
}

_GROUPS = ("paradigms", "env", "agent", "model", "site")


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open() as handle:
            return yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return {}


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.yaml") if not p.name.startswith("_"))


@dataclass(frozen=True)
class Environment:
    name: str
    offline_only: bool
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Environment:
        raw = _read_yaml(path)
        return cls(
            name=raw.get("name") or path.stem,
            offline_only=bool(raw.get("offline_only", False)),
            raw=raw,
        )


@dataclass(frozen=True)
class Agent:
    name: str
    algorithm: str
    hyperparameters: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Agent:
        raw = _read_yaml(path)
        return cls(
            name=path.stem,
            algorithm=raw.get("algorithm") or path.stem,
            hyperparameters={k: v for k, v in raw.items() if k != "algorithm"},
        )


@dataclass(frozen=True)
class Model:
    name: str
    architecture: str
    parameters: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Model:
        raw = _read_yaml(path)
        return cls(
            name=path.stem,
            architecture=raw.get("architecture") or path.stem,
            parameters={k: v for k, v in raw.items() if k != "architecture"},
        )


@dataclass(frozen=True)
class Experiment:
    name: str
    group: str
    path: Path
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path, root: Path) -> Experiment:
        relative = path.relative_to(root)
        return cls(
            name=str(relative.with_suffix("")),
            group=relative.parent.name or "ungrouped",
            path=path,
            raw=_read_yaml(path),
        )


@dataclass(frozen=True)
class Paradigm:
    name: str
    description: str
    allowed_agents: tuple[str, ...]
    forbidden_agents: tuple[str, ...]
    requires: Mapping[str, Any] = field(default_factory=dict, repr=False)
    forbids: Mapping[str, Any] = field(default_factory=dict, repr=False)
    raw: Mapping[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_file(cls, path: Path) -> Paradigm:
        raw = _read_yaml(path)
        constraints = raw.get("constraints") or {}
        return cls(
            name=path.stem,
            description=raw.get("description", ""),
            allowed_agents=tuple(raw.get("allowed_agents") or ()),
            forbidden_agents=tuple(raw.get("forbidden_agents") or ()),
            requires=constraints.get("requires") or {},
            forbids=constraints.get("forbids") or {},
            raw=raw,
        )

    def permits_agent(self, agent: Agent | str) -> bool:
        name = agent.algorithm if isinstance(agent, Agent) else agent
        if name in self.forbidden_agents:
            return False
        # An empty allow-list means nothing has been declared valid yet, which is
        # the current state of the supervised paradigm.
        return name in self.allowed_agents

    def permits_environment(self, environment: Environment) -> bool:
        required = self.requires.get("env.offline_only")
        if required is None:
            return True
        return bool(required) == environment.offline_only

    def field_enabled(self, name: str) -> bool:
        if name in self.forbids:
            return False
        flag = _ALLOW_FLAGS.get(name)
        if flag is not None and flag in self.raw:
            return bool(self.raw[flag])
        return True

    def disabled_reason(self, name: str) -> str | None:
        if self.field_enabled(name):
            return None
        if name in self.forbids:
            return f"{self.name} forbids {name} ({self.forbids[name]})"
        return f"{self.name} does not enable {name}"


class ConfigTree:
    """The `in/config/` directory, parsed into queryable objects."""

    def __init__(self, config_root: Path):
        self.config_root = Path(config_root)
        self.paradigms = {p.stem: Paradigm.from_file(p) for p in _yaml_files(self.config_root / "paradigms")}
        self.environments = {e.stem: Environment.from_file(e) for e in _yaml_files(self.config_root / "env")}
        self.agents = {a.stem: Agent.from_file(a) for a in _yaml_files(self.config_root / "agent")}
        self.models = {m.stem: Model.from_file(m) for m in _yaml_files(self.config_root / "model")}
        self.sites = tuple(s.stem for s in _yaml_files(self.config_root / "site"))

        experiment_root = self.config_root / "experiment"
        self.experiments: dict[str, Experiment] = {}
        if experiment_root.is_dir():
            for path in sorted(experiment_root.glob("**/*.yaml")):
                if path.name.startswith("_"):
                    continue
                experiment = Experiment.from_file(path, experiment_root)
                self.experiments[experiment.name] = experiment

    @classmethod
    def discover(cls, start: Path | None = None) -> ConfigTree:
        """Walk upward from `start` looking for an `in/config` directory."""
        current = Path(start or Path(__file__).resolve().parent)
        for candidate in (current, *current.parents):
            config_root = candidate / "in" / "config"
            if config_root.is_dir():
                return cls(config_root)
        raise FileNotFoundError(f"No in/config directory found above {current}")

    def agents_for(self, paradigm: str) -> list[Agent]:
        rules = self.paradigms[paradigm]
        return [a for a in self.agents.values() if rules.permits_agent(a)]

    def environments_for(self, paradigm: str) -> list[Environment]:
        rules = self.paradigms[paradigm]
        return [e for e in self.environments.values() if rules.permits_environment(e)]

    def field_enabled(self, paradigm: str, name: str) -> bool:
        return self.paradigms[paradigm].field_enabled(name)

    def experiments_in(self, group: str) -> list[Experiment]:
        return [e for e in self.experiments.values() if e.group == group]

    @property
    def groups(self) -> list[str]:
        return sorted({e.group for e in self.experiments.values()})


@dataclass
class Selection:
    """A chosen experiment plus the overrides the panel will apply to it."""

    experiment: str
    paradigm: str | None = None
    environment: str | None = None
    agent: str | None = None
    model: str | None = None
    site: str = "local"
    experiment_id: str | None = None
    seed: int | None = None
    total_timesteps: int | None = None
    intervals_count: int | None = None
    eval_episodes: int | None = None
    extra: list[str] = field(default_factory=list)

    _GROUP_FIELDS = (
        ("paradigm", "paradigm"),
        ("environment", "env"),
        ("agent", "agent"),
        ("model", "model"),
        ("site", "site"),
    )
    _SCALAR_FIELDS = (
        ("experiment_id", "experiment_id"),
        ("seed", "seed"),
        ("total_timesteps", "total_timesteps"),
        ("intervals_count", "intervals_count"),
        ("eval_episodes", "eval_episodes"),
    )

    def to_overrides(self, tree: ConfigTree | None = None) -> list[str]:
        """Render the selection as Hydra `key=value` strings.

        Passing `tree` drops overrides the chosen paradigm forbids, so the panel
        cannot submit a combination the pipeline would reject at startup.
        """
        overrides: list[str] = []
        for attribute, key in self._GROUP_FIELDS:
            value = getattr(self, attribute)
            if value is not None:
                overrides.append(f"{key}={value}")
        for attribute, key in self._SCALAR_FIELDS:
            value = getattr(self, attribute)
            if value is None:
                continue
            if tree is not None and self.paradigm and not tree.field_enabled(self.paradigm, key):
                continue
            overrides.append(f"{key}={value}")
        overrides.extend(self.extra)
        return overrides

    def command(
        self,
        tree: ConfigTree | None = None,
        python: str = "python",
        script: str = "run_pipeline.py",
        dry_run: bool = False,
    ) -> list[str]:
        argv = [python, script, self.experiment, *self.to_overrides(tree)]
        if dry_run:
            argv.append("dry_run=true")
        return argv

    def command_line(self, **kwargs: Any) -> str:
        return " ".join(self.command(**kwargs))
