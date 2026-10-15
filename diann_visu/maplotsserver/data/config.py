from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class AppConfig:
    data_roots: tuple["DataRoot", ...]
    pipeline_versions: tuple[str, ...]
    maplot_root: Path
    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = False

    @property
    def pipeline_version(self) -> str:
        return self.pipeline_versions[0]

    @property
    def pipeline_root(self) -> Path:
        return self.data_roots[0].path / self.pipeline_version

    @property
    def pipeline_roots(self) -> tuple[Path, ...]:
        return tuple(root.path / version for root in self.data_roots for version in self.pipeline_versions)

    @property
    def maplot_src(self) -> Path:
        return self.maplot_root / "src"

    @property
    def dash_plotly_upset_src(self) -> Path:
        return self.maplot_root.parent / "dash_plotly_upset" / "src"

    @classmethod
    def from_env(cls) -> "AppConfig":
        pipeline_versions = parse_pipeline_versions(
            os.environ.get("PIPELINE_VERSIONS") or os.environ.get("PIPELINE_VERSION"),
            default=("2_5_0", "2_6_0"),
        )
        config_path = Path(os.environ.get("MAPLOT_CONFIG", Path(__file__).with_name("config.toml"))).expanduser()
        data_roots = load_data_roots(config_path)
        return cls(
            data_roots=data_roots,
            pipeline_versions=pipeline_versions,
            maplot_root=Path(
                os.environ.get("MAPLOT_ROOT", "./diann_visu/post_run_reports")
            ).expanduser(),
            host=os.environ.get("DASH_HOST", "127.0.0.1"),
            port=int(os.environ.get("DASH_PORT", "8050")),
            debug=os.environ.get("DASH_DEBUG", "0") == "1",
        )


def parse_pipeline_versions(value: str | None, *, default: Iterable[str]) -> tuple[str, ...]:
    versions = tuple(part.strip() for part in (value or "").split(",") if part.strip())
    return versions or tuple(default)


@dataclass(frozen=True)
class DataRoot:
    key: str
    path: Path
    cluster: str


def load_data_roots(path: Path) -> tuple[DataRoot, ...]:
    """Load named DIA-NN roots; the first root remains the default/local root."""
    if not path.exists():
        return (DataRoot("local", Path(os.environ.get("DIANN_OUT_ROOT", "./diann_out")).expanduser(), "local"),)
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    roots = tuple(
        DataRoot(
            key=str(item["key"]),
            path=Path(str(item["path"])).expanduser(),
            cluster=str(item.get("cluster", item["key"])),
        )
        for item in payload.get("data_roots", [])
    )
    return roots or (DataRoot("local", Path("./diann_out"), "local"),)
