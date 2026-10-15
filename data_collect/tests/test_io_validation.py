from pathlib import Path

from rt_dataset.io import DatasetStore
from rt_dataset.schemas import assign_stable_ids, harmonize_schema
from rt_dataset.validation import validate_dataset


def test_parquet_round_trip(tmp_path: Path, identifications):
    data = assign_stable_ids(harmonize_schema(identifications))
    store = DatasetStore(tmp_path)
    store.save_predataset(data)
    loaded = store.load_predataset()
    assert loaded["dataset_row_hash"].tolist() == data["dataset_row_hash"].tolist()


def test_validation_detects_invalid_qvalue(identifications):
    data = assign_stable_ids(harmonize_schema(identifications))
    data.loc[data.index[0], "q_value"] = 2.0
    report = validate_dataset(data)
    assert report["checks"]["invalid_q_value"]["count"] == 1
    assert not report["valid"]
