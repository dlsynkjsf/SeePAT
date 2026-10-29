"""Optional execution controls, separate from frozen scientific training options."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingRuntimeOptions:
    clip_cache_dir: Path | None = None
    feature_cache_dir: Path | None = None
    persistent_train_workers: bool = False

    def validate(self) -> None:
        for name in ("clip_cache_dir", "feature_cache_dir"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, Path):
                raise TypeError(f"{name} must be a Path or null")
        if type(self.persistent_train_workers) is not bool:
            raise TypeError("persistent_train_workers must be boolean")

    def record(self) -> dict:
        self.validate()
        return {
            "clip_cache_dir": str(self.clip_cache_dir) if self.clip_cache_dir else None,
            "feature_cache_dir": str(self.feature_cache_dir) if self.feature_cache_dir else None,
            "persistent_train_workers": self.persistent_train_workers,
        }


def load_runtime_options(raw: dict | None, project_root: Path = Path('.')):
    if raw is None:
        return TrainingRuntimeOptions()
    if not isinstance(raw, dict):
        raise TypeError("Training runtime settings must be a mapping")
    settings = dict(raw)
    for key in ("clip_cache_dir", "feature_cache_dir"):
        if settings.get(key) is not None:
            path = Path(settings[key])
            settings[key] = path if path.is_absolute() else project_root / path
    result = TrainingRuntimeOptions(**settings)
    result.validate()
    return result
