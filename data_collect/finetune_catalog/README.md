# Fine-tuning dataset catalog

`datasets.yaml` is the local registry of Carafe-derived training sources.
Each source has a permanent ID, original experiment name, organism, setup
metadata and comment. Crucially, it records both the **upstream DIA-NN Step-2
report** and the **Carafe job/export** generated from it. A build selects
registered source IDs and produces one reproducible dataset for both model
families. Copy `datasets.example.yaml` to `datasets.yaml` before first use.
The local registry and generated builds are intentionally excluded from Git
because they contain machine paths and potentially private experiment names.

## Source files produced by Carafe

Carafe itself creates the training tables:

- `rt_train_data.tsv` from `AIGear.generate_rt_train_data()` — the shared RT
  source for **both Carafe2 and DIA-NN**;
- `psm_pdv.txt`, `fragment_intensity_df.tsv`, and optionally
  `fragment_intensity_valid.tsv` — required only for Carafe2 MS2 fine-tuning.

## Commands

```bash
cd /path/to/carafe2-workspace/ma/data_collect

python3 fine_tune_catalog.py \
  --catalog finetune_catalog/datasets.yaml list

# After preparing/running a new Carafe job from another DIA-NN search result,
# print a complete YAML template; fill in its permanent ID and scientific metadata.
python3 fine_tune_catalog.py \
  --catalog finetune_catalog/datasets.yaml discover-job \
  --job-dir /path/to/carafe/<profile>/<job>

python3 fine_tune_catalog.py \
  --catalog finetune_catalog/datasets.yaml validate \
  --build celegans_exploris_44min_rtms2_v1

python3 fine_tune_catalog.py \
  --catalog finetune_catalog/datasets.yaml build \
  --build celegans_exploris_44min_rtms2_v1
```

The resulting directory has this layout:

```text
<output_root>/<build-id>/
├── carafe2_input/       # RT + safely re-indexed MS2 tables
├── diann_rt_input/      # byte-identical rt_train_data.tsv + provenance
├── build_manifest.json  # source metadata, checksums, normalisation policy
└── README.md
```

Use `carafe2_input` with Carafe2, and `diann_rt_input` with the DIA-NN
fine-tuning scripts. To add a new source, append a new permanent entry under
`datasets`, then define a new build ID; existing builds remain reproducible.

## Add a dataset from DIA-NN Step 2 and mzML files

The catalog does **not** derive training tables directly from a DIA-NN report:
Carafe needs the Step-2 report, its matching training mzML files, and the
matching FASTA to extract and validate RT/MS2 training targets. The standard
pipeline command prepares this immutable Carafe job from an existing DIA-NN
experiment (and discovers the exact Step-2 report, FASTA and training mzMLs
from the recorded command):

```bash
cd /path/to/carafe2-workspace/ma

bash diaNN_pipline/prepare_carafe_job.sh \
  --source-exp <dia-nn-experiment-name> \
  --pipeline-version 2_5_0 \
  --profile balanced \
  --name <permanent-source-id> \
  --set CARAFE_TUNE_TYPE=all
```

Run that job on the GPU machine (this creates `carafe_output/`):

```bash
bash diaNN_pipline/run_carafe_profile.sh \
  --experiment <dia-nn-experiment-name> \
  --pipeline-version 2_5_0 \
  --job balanced/<permanent-source-id> \
  --fine-tune-only
```

For a dataset stored outside a normal DIA-NN experiment, use the
dedicated external-source wrapper. It stages a normalised private report,
symlinks every mzML below the supplied directory, writes the compatibility
contract and records checksums:

```bash
cd /path/to/carafe2-workspace/ma

bash data_collect/prepare_external_carafe_source.sh \
  --id <permanent-source-id> \
  --report /path/to/tune_search_report.parquet \
  --mzml-dir /path/to/training_mzmls \
  --fasta /path/to/exact-search.fasta \
  --experiment-name '<original experiment name>'

# Then run the printed job path on the GPU host (the job README contains the exact command).
```

The default Carafe configuration is the current semi-specific Heavy-Dimethyl
configuration. For another search setup, use a matching config or explicit
`--set CARAFE_...=...` overrides before running it. Do not combine runs from
different gradients until their `rt_max` values have passed the catalog's
strict normalisation check.

Once `carafe_output/` exists, the following command prints a ready-to-paste
catalog entry. Fill in its ID and scientific metadata, append it under
`datasets` in `datasets.yaml`, and create a new build that lists all source
IDs to combine:

```bash
python3 fine_tune_catalog.py \
  --catalog finetune_catalog/datasets.yaml discover-job \
  --job-dir /path/to/carafe/<profile>/<permanent-source-id>
```

Finally validate and build the new combined ID. A build is immutable: changing
its source list requires a new build ID.
