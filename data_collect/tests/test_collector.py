from pathlib import Path

import pandas as pd

from rt_dataset.collectors import RTDatasetCollector
from rt_dataset.config import CollectConfig, ReaderConfig
from rt_dataset.io import DatasetStore
from rt_dataset.readers import GenericTabularReader


def test_incremental_collection_skips_imported_file(tmp_path: Path):
    source = tmp_path / "report.csv"
    pd.DataFrame(
        {
            "Sequence": ["PEPTIDE"],
            "Run": ["run_a"],
            "RT": [12.0],
            "Q": [0.001],
            "Decoy": [False],
        }
    ).to_csv(source, index=False)
    config = CollectConfig(
        output_dir=str(tmp_path / "output"),
        reader=ReaderConfig(
            column_mapping={
                "sequence": "Sequence",
                "raw_file_name": "Run",
                "observed_rt": "RT",
                "q_value": "Q",
                "is_decoy": "Decoy",
            }
        ),
    )
    store = DatasetStore(Path(config.output_dir))
    collector = RTDatasetCollector(GenericTabularReader(config.reader), store=store, config=config)
    first = collector.collect([source])
    second = collector.collect([source])
    assert len(first) == len(second) == 1
    assert len(store.load_manifest()) == 1
