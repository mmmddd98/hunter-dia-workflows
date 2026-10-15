from pathlib import Path

import pandas as pd
import pytest

from rt_dataset.config import ReaderConfig
from rt_dataset.metadata import RawFileMetadataMapper
from rt_dataset.readers import GenericTabularReader


def test_column_mapping(tmp_path: Path):
    path = tmp_path / "report.csv"
    pd.DataFrame({"Stripped.Sequence": ["PEPTIDE"], "Run": ["run_a"], "RT": [12.0]}).to_csv(path, index=False)
    reader = GenericTabularReader(
        ReaderConfig(
            column_mapping={"sequence": "Stripped.Sequence", "raw_file_name": "Run", "observed_rt": "RT"}
        )
    )
    result = reader.read(path)
    assert result.loc[0, "sequence"] == "PEPTIDE"
    assert result.loc[0, "raw_file_name"] == "run_a"


def test_missing_required_column_raises(tmp_path: Path):
    path = tmp_path / "report.csv"
    pd.DataFrame({"Run": ["run_a"], "RT": [12.0]}).to_csv(path, index=False)
    reader = GenericTabularReader(ReaderConfig(column_mapping={"raw_file_name": "Run", "observed_rt": "RT"}))
    with pytest.raises(ValueError, match="sequence"):
        reader.read(path)


def test_metadata_mapping_and_unmapped_report():
    mapping = pd.DataFrame({"raw_file_name": ["run_a"], "project_id": ["project_1"], "hunter_status": ["HUNTER"]})
    data = pd.DataFrame({"raw_file_name": ["run_a", "run_b"]})
    mapped, report = RawFileMetadataMapper(mapping).map_metadata(data)
    assert mapped.loc[0, "project_id"] == "project_1"
    assert mapped.loc[1, "hunter_status"] == "UNKNOWN"
    assert report.unmapped_raw_files == ["run_b"]
