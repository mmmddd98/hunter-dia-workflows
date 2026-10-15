#!/usr/bin/env python3
"""Convert CARAFE's native Parquet library to the DIA-NN 2.x Parquet schema.

The converter streams record batches, so it is suitable for multi-billion-row
CARAFE libraries.  It deliberately keeps the source library untouched.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="CARAFE raw Parquet library")
    parser.add_argument("--output", required=True, type=Path, help="DIA-NN 2.x-compatible Parquet library")
    parser.add_argument("--reference", required=True, type=Path, help="Known-good DIA-NN 2.x library; supplies the exact schema")
    parser.add_argument("--batch-size", type=int, default=65_536)
    return parser.parse_args()


def diann_modified_sequence(carafe_peptide: str) -> str:
    """Convert CARAFE's _PEP[UniMod:n]_ spelling to DIA-NN's PEP(UniMod:n)."""
    sequence = carafe_peptide.strip("_")
    return sequence.replace("[UniMod:", "(UniMod:").replace("]", ")")


def protein_ids(carafe_ids: str) -> list[str]:
    ids: list[str] = []
    for entry in carafe_ids.split(";"):
        parts = entry.split("|")
        accession = parts[1] if len(parts) >= 2 else entry
        if accession and accession not in ids:
            ids.append(accession)
    return ids


def build_table(batch: pa.RecordBatch, schema: pa.Schema, previous_precursor: str | None) -> tuple[pa.Table, str | None]:
    raw = batch.to_pydict()
    columns: dict[str, list[object]] = {field.name: [] for field in schema}
    last_precursor = previous_precursor

    for values in zip(*(raw[name] for name in batch.schema.names)):
        row = dict(zip(batch.schema.names, values))
        modified = diann_modified_sequence(row["ModifiedPeptide"])
        stripped = row["StrippedPeptide"]
        charge = int(row["PrecursorCharge"])
        precursor = f"{modified}{charge}"
        ids = protein_ids(row["ProteinID"] or "")
        proteins = ";".join(ids)
        first_fragment = precursor != last_precursor

        values_by_column = {
            "Precursor.Id": precursor,
            "Modified.Sequence": modified,
            "Stripped.Sequence": stripped,
            "Precursor.Charge": charge,
            "Proteotypic": int(len(ids) == 1),
            "Decoy": int(row["Decoy"]),
            # CARAFE does not record protein-terminal status.  Zero is the
            # conservative DIA-NN value (not a terminal peptide).
            "N.Term": 0,
            "C.Term": 0,
            "RT": float(row["Tr_recalibrated"]),
            "IM": float(row["IonMobility"]),
            # This is a predicted target library, not an empirical library.
            "Q.Value": 0.0,
            "Peptidoform.Q.Value": 0.0,
            "PTM.Site.Confidence": 1.0,
            "PG.Q.Value": 0.0,
            "Precursor.Mz": float(row["PrecursorMz"]),
            "Product.Mz": float(row["FragmentMz"]),
            "Relative.Intensity": float(row["RelativeIntensity"]),
            "Fragment.Type": row["FragmentType"] or "",
            "Fragment.Charge": int(row["FragmentCharge"]),
            "Fragment.Series.Number": int(row["FragmentNumber"]),
            "Fragment.Loss.Type": row["FragmentLossType"] or "noloss",
            "Fragment.Score": 0.0,
            "Exclude.From.Quant": 0,
            "Protein.Ids": proteins,
            # DIA-NN can replace these annotations from the supplied FASTA.
            "Protein.Group": proteins,
            "Protein.Names": proteins,
            "Genes": proteins,
            # Bit 0: fragment enabled; bit 4: first fragment of precursor.
            "Flags": 1 | (16 if first_fragment else 0),
            # Source.Id is relevant for supplied decoys only; CARAFE creates targets.
            "Source.Id": "",
        }
        for name in columns:
            columns[name].append(values_by_column[name])
        last_precursor = precursor

    arrays = [pa.array(columns[field.name], type=field.type) for field in schema]
    return pa.Table.from_arrays(arrays, schema=schema), last_precursor


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.output}")
    partial_output = args.output.with_name(args.output.name + ".partial")
    if partial_output.exists():
        print(f"removing incomplete prior conversion: {partial_output}", flush=True)
        partial_output.unlink()

    reference_schema = pq.ParquetFile(args.reference).schema_arrow
    expected = {
        "Precursor.Id", "Modified.Sequence", "Stripped.Sequence", "Precursor.Charge",
        "Proteotypic", "Decoy", "N.Term", "C.Term", "RT", "IM", "Q.Value",
        "Peptidoform.Q.Value", "PTM.Site.Confidence", "PG.Q.Value", "Precursor.Mz",
        "Product.Mz", "Relative.Intensity", "Fragment.Type", "Fragment.Charge",
        "Fragment.Series.Number", "Fragment.Loss.Type", "Fragment.Score",
        "Exclude.From.Quant", "Protein.Ids", "Protein.Group", "Protein.Names", "Genes",
        "Flags", "Source.Id",
    }
    if set(reference_schema.names) != expected:
        raise SystemExit("Reference schema is not the expected DIA-NN 2.x library schema")

    source = pq.ParquetFile(args.input)
    required_source = {
        "Decoy", "FragmentCharge", "FragmentLossType", "FragmentMz", "FragmentNumber",
        "FragmentType", "IonMobility", "ModifiedPeptide", "PrecursorCharge", "PrecursorMz",
        "ProteinID", "RelativeIntensity", "StrippedPeptide", "Tr_recalibrated",
    }
    missing = required_source - set(source.schema_arrow.names)
    if missing:
        raise SystemExit(f"Input is not the expected CARAFE raw schema; missing {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    last_precursor: str | None = None
    with pq.ParquetWriter(partial_output, reference_schema, compression="zstd", use_dictionary=True) as writer:
        for batch in source.iter_batches(batch_size=args.batch_size):
            table, last_precursor = build_table(batch, reference_schema, last_precursor)
            writer.write_table(table, row_group_size=args.batch_size)
            written += table.num_rows
            if written % (args.batch_size * 100) == 0:
                print(f"converted {written:,}/{source.metadata.num_rows:,} rows", flush=True)
    os.replace(partial_output, args.output)
    print(f"wrote {written:,} rows to {args.output}")


if __name__ == "__main__":
    main()
