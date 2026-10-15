#!/usr/bin/env python3
"""Create a private Carafe report with canonical DIA-NN column names."""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CANONICAL_SCAN_COLUMN = "MS2.Scan"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"Refusing to overwrite an existing report destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    names = schema.names
    if CANONICAL_SCAN_COLUMN in names:
        renamed_index = None
        target_schema = schema
    else:
        matches = [i for i, name in enumerate(names) if name.casefold() == CANONICAL_SCAN_COLUMN.casefold()]
        if len(matches) != 1:
            found = ", ".join(name for name in names if "scan" in name.casefold()) or "none"
            raise SystemExit(f"Cannot normalize {source}: expected exactly one {CANONICAL_SCAN_COLUMN} variant; found: {found}.")
        renamed_index = matches[0]
        fields = list(schema)
        fields[renamed_index] = fields[renamed_index].with_name(CANONICAL_SCAN_COLUMN)
        target_schema = pa.schema(fields, metadata=schema.metadata)

    # The temporary file and final rename reside in the destination directory.
    # The destination has been required not to exist, so no existing report is
    # ever renamed or replaced by this utility.
    with tempfile.NamedTemporaryFile(prefix=".diann_step2_report.", suffix=".parquet", dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with pq.ParquetWriter(temporary, target_schema, compression="zstd") as writer:
            for batch in parquet.iter_batches(batch_size=100_000):
                writer.write_batch(pa.RecordBatch.from_arrays(list(batch), schema=target_schema))
        os.rename(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    if renamed_index is None:
        print(f"Copied report with canonical {CANONICAL_SCAN_COLUMN}: {destination}")
    else:
        print(f"Normalized DIA-NN column {names[renamed_index]} -> {CANONICAL_SCAN_COLUMN}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
