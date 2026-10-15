from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

from .aggregation import AggregationConfig, aggregate_identifications
from .collectors import RTDatasetCollector
from .config import CollectConfig, FilterConfig, load_config
from .filtering import RTDatasetFilter
from .io import DatasetStore
from .metadata import RawFileMetadataMapper
from .qc import create_qc_plots
from .readers import GenericTabularReader
from .rt_conversion import ChronologerHITransformer, IRTTransformer, save_calibration_coefficients
from .validation import validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    command = args.command
    if command == "collect":
        collect_command(Path(args.config))
    elif command == "map-metadata":
        map_metadata_command(Path(args.config))
    elif command == "filter":
        filter_command(Path(args.config))
    elif command == "aggregate":
        aggregate_command(Path(args.config))
    elif command == "convert-irt":
        convert_irt_command(Path(args.config))
    elif command == "convert-hi":
        convert_hi_command(Path(args.config))
    elif command == "validate":
        validate_command(Path(args.dataset), Path(args.output) if args.output else None)
    else:
        parser.error("A command is required")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rt-dataset")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("collect", "map-metadata", "filter", "aggregate", "convert-irt", "convert-hi"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset", required=True)
    validate.add_argument("--output")
    return parser


def collect_command(path: Path) -> None:
    config = load_config(path, CollectConfig)
    metadata = read_table(Path(config.metadata_path)) if config.metadata_path else None
    mapper = RawFileMetadataMapper(metadata, allow_regex=config.metadata_regex) if metadata is not None else None
    store = DatasetStore(Path(config.output_dir))
    collector = RTDatasetCollector(
        GenericTabularReader(config.reader),
        metadata_mapper=mapper,
        store=store,
        config=config,
    )
    dataset = collector.collect(expand_input_paths(config.input_paths))
    report = validate_dataset(dataset)
    store.save_validation_report(report)
    print(f"Pre-dataset: {store.predataset_path} ({len(dataset):,} rows)")


def map_metadata_command(path: Path) -> None:
    config = yaml.safe_load(path.read_text()) or {}
    dataset = read_table(Path(config["dataset"]))
    mapping = read_table(Path(config["metadata_path"]))
    mapped, report = RawFileMetadataMapper(mapping, allow_regex=bool(config.get("metadata_regex"))).map_metadata(dataset)
    mapped.to_parquet(config["output"], index=False)
    Path(config.get("report", "metadata_mapping_report.json")).write_text(json.dumps(asdict(report), indent=2))


def filter_command(path: Path) -> None:
    payload = yaml.safe_load(path.read_text()) or {}
    config = FilterConfig(**payload.get("filters", {}))
    filtered, report = RTDatasetFilter().apply(read_table(Path(payload["dataset"])), config)
    filtered.to_parquet(payload["output"], index=False)
    Path(payload.get("report", "filter_report.json")).write_text(json.dumps(report, indent=2, default=str))


def aggregate_command(path: Path) -> None:
    payload = yaml.safe_load(path.read_text()) or {}
    config = AggregationConfig(**payload.get("aggregation", {}))
    aggregate_identifications(read_table(Path(payload["dataset"])), config).to_parquet(payload["output"], index=False)


def convert_irt_command(path: Path) -> None:
    payload = yaml.safe_load(path.read_text()) or {}
    dataset = read_table(Path(payload["dataset"]))
    calibration = read_table(Path(payload["calibration"]))
    transformer = IRTTransformer(**payload.get("transformer", {})).fit(calibration)
    transformer.transform(dataset).to_parquet(payload["output"], index=False)
    save_calibration_coefficients(transformer, Path(payload.get("coefficients", "irt_coefficients.json")))


def convert_hi_command(path: Path) -> None:
    payload = yaml.safe_load(path.read_text()) or {}
    dataset = read_table(Path(payload["dataset"]))
    transformer = ChronologerHITransformer(
        Path(payload["script_path"]),
        python_executable=payload.get("python_executable", "python"),
        extra_args=payload.get("extra_args"),
    ).fit(dataset)
    transformer.transform(dataset).to_parquet(payload["output"], index=False)


def validate_command(dataset_path: Path, output: Path | None) -> None:
    dataset = read_table(dataset_path)
    report = validate_dataset(dataset)
    output = output or dataset_path.with_name("validation_report.json")
    output.write_text(json.dumps(report, indent=2, default=str))
    create_qc_plots(dataset, output.parent / "qc_plots")
    print(f"Validation report: {output}")


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".txt"} else ",")


def expand_input_paths(patterns: list[str]) -> list[Path]:
    paths: set[Path] = set()
    for pattern in patterns:
        candidate = Path(pattern)
        if candidate.is_file():
            paths.add(candidate)
        elif candidate.is_dir():
            paths.update(path for path in candidate.rglob("*") if path.suffix.lower() in {".csv", ".tsv", ".txt", ".parquet"})
        else:
            paths.update(Path().glob(pattern))
    return sorted(paths)


if __name__ == "__main__":
    raise SystemExit(main())
