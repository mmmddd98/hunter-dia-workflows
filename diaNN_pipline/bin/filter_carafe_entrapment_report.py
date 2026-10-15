#!/usr/bin/env python3
"""Remove target report rows whose peptide is not present in the original FASTA.

Entrapment FASTAs reuse the original protein identifiers but contain shuffled
sequences.  Protein IDs alone therefore cannot distinguish the two cases; the
reported stripped peptide is checked against the original protein sequence.
Decoy rows are retained because they are DIA-NN search metadata, not
entrapment target identifications used for Carafe training.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def fasta_sequences(path: Path) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_ids: list[str] = []
    chunks: list[str] = []

    def flush() -> None:
        if not current_ids:
            return
        sequence = "".join(chunks).upper()
        for identifier in current_ids:
            sequences[identifier] = sequence

    for line in path.read_text(errors="replace").splitlines():
        if line.startswith(">"):
            flush()
            chunks = []
            header = line[1:].split(None, 1)[0]
            current_ids = [header]
            if "|" in header:
                parts = header.split("|")
                if len(parts) >= 2:
                    current_ids.append(parts[1])
        else:
            chunks.append(line.strip())
    flush()
    return sequences


def identifiers(value: object) -> list[str]:
    if value is None:
        return []
    result: list[str] = []
    for item in re.split(r"[;,]", str(value)):
        item = item.strip()
        if not item:
            continue
        result.append(item)
        if "|" in item:
            parts = item.split("|")
            if len(parts) >= 2:
                result.append(parts[1])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--stats", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"Refusing to overwrite an existing report destination: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    sequences = fasta_sequences(args.fasta)
    parquet = pq.ParquetFile(source)
    schema = parquet.schema_arrow
    names = schema.names
    canonical_scan_column = "MS2.Scan"
    if canonical_scan_column in names:
        target_schema = schema
    else:
        matches = [i for i, name in enumerate(names) if name.casefold() == canonical_scan_column.casefold()]
        if len(matches) != 1:
            found = ", ".join(name for name in names if "scan" in name.casefold()) or "none"
            raise SystemExit(
                f"Cannot normalize {source}: expected exactly one {canonical_scan_column} variant; found: {found}."
            )
        fields = list(schema)
        fields[matches[0]] = fields[matches[0]].with_name(canonical_scan_column)
        target_schema = pa.schema(fields, metadata=schema.metadata)

    input_rows = 0
    kept_rows = 0
    removed = 0
    retained_decoys = 0
    missing_protein_mapping = 0

    with tempfile.NamedTemporaryFile(
        prefix=".diann_step2_report.filtered.", suffix=".parquet", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with pq.ParquetWriter(temporary, target_schema, compression="zstd") as writer:
            for batch in parquet.iter_batches(batch_size=100_000):
                columns = {name: batch.column(i) for i, name in enumerate(batch.schema.names)}
                if "Stripped.Sequence" not in columns or "Protein.Ids" not in columns:
                    raise SystemExit("Report lacks Stripped.Sequence or Protein.Ids")
                peptide_values = columns["Stripped.Sequence"].to_pylist()
                protein_values = columns["Protein.Ids"].to_pylist()
                decoy_values = columns.get("Decoy")
                decoys = decoy_values.to_pylist() if decoy_values is not None else [0] * len(peptide_values)
                keep: list[bool] = []
                for peptide, protein_ids, decoy in zip(peptide_values, protein_values, decoys):
                    input_rows += 1
                    if decoy not in (0, 0.0, False, None):
                        retained_decoys += 1
                        keep.append(True)
                        continue
                    peptide_text = str(peptide or "").upper()
                    protein_identifiers = identifiers(protein_ids)
                    mapped_sequences = [sequences[item] for item in protein_identifiers if item in sequences]
                    if not mapped_sequences:
                        missing_protein_mapping += 1
                    matched = bool(peptide_text) and any(peptide_text in sequence for sequence in mapped_sequences)
                    keep.append(matched)
                filtered = batch.filter(pa.array(keep))
                kept_rows += filtered.num_rows
                removed += len(keep) - sum(keep)
                writer.write_batch(pa.RecordBatch.from_arrays(list(filtered), schema=target_schema))
        os.rename(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    stats = {
        "source_report": str(source),
        "filtered_report": str(destination),
        "original_fasta": str(args.fasta.resolve()),
        "input_rows": input_rows,
        "kept_rows": kept_rows,
        "excluded_entrapment_target_rows": removed,
        "retained_decoy_rows": retained_decoys,
        "target_rows_without_original_protein_mapping": missing_protein_mapping,
    }
    if args.stats:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"Filtered {removed} entrapment-only target rows; kept {kept_rows}/{input_rows}: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
