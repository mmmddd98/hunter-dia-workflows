# RT Dataset

`rt-dataset` is a modular Python package for data collection, metadata mapping,
filtering, aggregation, validation, and retention-time target preparation. It
creates traceable, model-ready datasets but does not train prediction models.

## Design

The workflow separates four concerns:

1. **Data collection** reads heterogeneous search-result tables through a
   configurable reader interface.
2. **Metadata mapping** joins explicit project and sample metadata to raw-file
   names. No HUNTER state is inferred unless the mapping table or a configured
   rule provides it.
3. **Pre-dataset creation** applies only the initial q-value/decoy policy,
   assigns deterministic IDs, and records every imported source in a manifest.
4. **Final dataset creation** applies model-specific filters, aggregation, and
   optional iRT or Hydrophobic Index conversion.

The implementation uses a canonical schema, abstract reader/transformer
interfaces, typed configuration dataclasses, deterministic hashes, Parquet
storage, explicit validation, and a modular and maintainable codebase.

## Installation

```bash
cd data_collect
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test]"
```

Python 3.11 or newer is required.

## Package structure

```text
rt_dataset/
├── aggregation.py
├── cli.py
├── collectors.py
├── config.py
├── filtering.py
├── harmonization.py
├── io.py
├── metadata.py
├── qc.py
├── readers/
│   ├── base.py
│   └── generic_tabular.py
├── rt_conversion.py
├── schemas.py
└── validation.py
```

## End-to-end workflow

### 1. Configure data collection

Edit [`examples/collect.yaml`](examples/collect.yaml). Column mapping is
canonical-name to source-name:

```yaml
reader:
  column_mapping:
    sequence: Stripped.Sequence
    raw_file_name: Run
    q_value: Q.Value
    observed_rt: RT
```

The generic reader supports CSV, TSV, TXT, and Parquet. Unknown input columns
are preserved as JSON in `source_metadata` when
`preserve_unknown_columns: true`.

### 2. Collect the pre-dataset

```bash
rt-dataset collect --config examples/collect.yaml
```

This writes:

```text
predataset.parquet
manifest.parquet
raw_file_metadata.parquet
unmapped_raw_files.csv
validation_report.json
```

Incremental collection checks SHA-256 source checksums and skips files already
listed in the manifest. `dataset_row_hash`, `source_file_id`, `raw_file_id`,
`precursor_id`, and `peptidoform_id` are deterministic.

### 3. Generate a final dataset

```bash
rt-dataset filter --config examples/filter.yaml
rt-dataset aggregate --config examples/aggregate.yaml
```

The filter report records rows before and after every active filter plus
sequence, peptidoform, precursor, raw-file, project, class, modification, and
RT summaries. Scientific thresholds are configuration choices rather than
hidden defaults.

Aggregation can produce precursor or peptidoform tables using best-score,
median, weighted-median, mean, or intensity-weighted RT. By default,
identifications are aggregated within each raw file first, preventing repeated
PSMs from over-weighting common peptides.

### 4. Create iRT targets

The calibration table must contain the configured run ID, observed RT, and
reference iRT columns. Calibration and evaluation rows should be selected
separately by the caller.

```bash
rt-dataset convert-irt --config examples/irt.yaml
```

Linear, robust Theil-Sen, and spline calibration are available. Calibration
coefficients are saved separately and original RT columns are retained.

### 5. Run Chronologer conversion

Create a YAML file containing:

```yaml
dataset: examples/output/final_peptidoforms.parquet
script_path: /path/to/Chronologer/convert.py
output: examples/output/final_peptidoforms_hi.parquet
python_executable: python
extra_args: []
```

Then run:

```bash
rt-dataset convert-hi --config chronologer.yaml
```

The wrapper passes `--input` and `--output` CSV paths to the external script,
captures stdout/stderr, validates stable IDs and `observed_hi`, preserves
original RT, and records a checksum of the conversion script. Adapt the
external command arguments if the installed Chronologer version uses a
different CLI contract.

### Mathematical run-to-reference RT alignment

`rt_alignment_methods.py` provides one common `fit(source_rt, reference_rt)` /
`transform(source_rt)` interface for affine, relative-gradient, LOWESS,
isotonic, monotonic-PCHIP, official Calib-RT, Chronologer-style KDE-ridge,
and external Chronologer hydrophobic-index mappings:

```python
from rt_alignment import create_alignment

aligner = create_alignment("monotonic_spline", n_bins=100)
aligner.fit(anchor_rt_source, anchor_rt_reference)
rt_aligned_minutes = aligner.transform(all_rt_source)
```

The learned source-to-reference mapping is called **RT alignment** (or RT
calibration). Dividing the aligned reference RT by a fixed duration for a
dimensionless neural-network target is a separate **target normalization**
step. Relative-gradient scaling is the one method in this module that is
normalization by itself. `ChronologerHIAlignment` is alignment because it maps
each run to one external HI coordinate; scaling that HI for neural-network
training remains target normalization. Calib-RT is an optional dependency and
can be enabled with `pip install -e ".[calibrt]"`. The HI benchmark expects the
official Chronologer checkout next to the workspace, overridable via
`CHRONOLOGER_ROOT`.

### 6. Validate and create QC plots

```bash
rt-dataset validate \
  --dataset examples/output/final_peptidoforms_irt.parquet \
  --output examples/output/validation_report.json
```

Validation checks missing and duplicate IDs, invalid q-values/RTs/sequences,
ambiguous metadata, inconsistent gradients, malformed modification positions,
and large calibration residuals. QC plots are exported as PNG and PDF.

## Metadata mapping

`RawFileMetadataMapper` supports exact matching and optional full regular
expressions. It reports unmapped files, duplicate mapping rows, and ambiguous
matches. Unmapped HUNTER status is `UNKNOWN`.

```python
mapping = pd.read_csv("raw_file_metadata.csv")
mapper = RawFileMetadataMapper(mapping, allow_regex=False)
mapped, report = mapper.map_metadata(identifications)
```

## Extending readers and targets

Implement `SearchResultReader.read()` for another search engine and return
canonical column names. Implement `RTTargetTransformer.fit()`, `transform()`,
and `inverse_transform()` for another target system. The collector and storage
layers do not depend on one search-engine format or RT model.

## Tests

```bash
pytest
```

Synthetic fixtures cover deterministic IDs, column mapping, filtering,
metadata conflicts, aggregation, iRT calibration, the Chronologer subprocess
contract, missing columns, and Parquet round trips.
