# maplot server

This server provides interactive quality-control views used in the thesis to
compare DIA-NN searches, inspect N-terminal modification groups, and review
run-level metrics. Its interactive UpSet, sequence-logo, and Temple plots use
the separate `dash-plotly-upset`, `pydaglogo`, and `plotly-temple-plot`
packages. These packages are not bundled in this repository; obtain and
configure them independently. The UpSet package is required to start the
server. The sequence-logo and Temple plot packages provide optional features.

## Install and run

Use Python 3.11 or newer. From the `hunter-dia-workflows/` repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r diann_visu/maplotsserver/requirements.txt
python -m pip install -e diann_visu/post_run_reports
python -m pip install "git+https://github.com/mmmddd98/dash-plotly-upset.git"
```

The required [dash-plotly-upset](https://github.com/mmmddd98/dash-plotly-upset)
package is maintained separately. Optional sequence-logo and Temple plots use
[pydaglogo](https://github.com/mmmddd98/pydaglogo) and
[plotly-temple-plot](https://github.com/mmmddd98/plotly-temple-plot); follow their
installation instructions to enable those features. Access to the component
repositories may be required.

Copy `diann_visu/maplotsserver/data/config.toml` to a local, untracked file and
set its `data_roots` paths to your DIA-NN output roots. Each root contains the
version folders, for example `2_5_0/`.

```bash
mkdir -p local_analysis
cp diann_visu/maplotsserver/data/config.toml local_analysis/maplot.toml
# Edit local_analysis/maplot.toml before starting the server.
export MAPLOT_CONFIG="$PWD/local_analysis/maplot.toml"
export MAPLOT_ROOT="$PWD/diann_visu/post_run_reports"
python diann_visu/maplotsserver/app.py
```

Open <http://127.0.0.1:8050>. The working directory for these commands is the
repository root; the paths in the example TOML are resolved from that directory.
`PIPELINE_VERSIONS` selects comma-separated version folders (default:
`2_5_0,2_6_0`). `DASH_HOST`, `DASH_PORT` and `DASH_DEBUG` override server settings.

## Server plots

The MAPLOT server provides interactive views for inspecting DIA-NN search results and N-terminal modification evidence.

### Dataset overview

Shows all available experiments and links to the plot pages for each dataset. The overview also reports basic metadata such as number of searches, organism, experiment type, and whether entrapment analysis is available.

Restricted-layer DIA-NN variants are detected below
`05_quant/tuned_restricted_layers/<profile>/lr_<learning-rate>/`.  Each of
these directories is shown as exactly one search (for example, `Restricted
layers · balanced · LR 0.00011`).  Report-like files written below an `eval/`
subdirectory are treated as artefacts of that same search, not additional
runs.  If only such a nested report is present while a run is still being
materialised, it is used as a temporary loading fallback.

External-model searches are detected below
`external_searches/<model-id>/<run-id>/05_quant/`. They are shown inside the
normal experiment with the readable label `External model · <model-id> ·
<run-id>` while retaining the storage path as the internal search id.

### QC overview

Summarizes basic run-level quality metrics across searches. This page is useful for quickly comparing searches and spotting unusual runs or parameter settings.

### Intensity distribution

Shows precursor/intensity distributions across searches. This helps compare signal distributions and identify searches with unusually low or shifted intensity profiles.

### N-termini overview

Shows the abundance and specificity of detected N-terminal evidence, including modification classes such as dimethylated, acetylated, and free N-termini.

### Mixed-sample comparison

For supported datasets classified as `Mixed`, this page compares sample groups across searches. It presents organism-resolved identifications, detection overlap, abundance relationships, and fold-change distributions. Group annotations are read from a tab-separated sample annotation file in the experiment directory.

### Interactive UpSet plot

Compares identified entities across searches. The entity level can be changed, for example precursor, N-terminal peptide, N-terminal site, or protein. Intersections can be clicked to inspect the underlying items. Additional detail plots summarize selected items by specificity, N-terminal modification, RT, mass error, peptide properties, and sequence/motif context.

### Entrapment FDR

Uses ENT_ entrapment proteins to estimate empirical false discovery behavior. The page compares target and entrapment IDs across selected aggregation levels and DIA-NN Q.Value thresholds. N-terminal peptide and site levels can be split by peptide evidence specificity, while precursor and protein levels are shown as combined evidence.

### RAM usage

Displays memory usage logs from pipeline runs. It shows RAM over time and summary views for individual steps, helping identify expensive or unstable pipeline stages.

### Live run monitor

Tracks currently running experiments using progress files, PID status, RAM logs, and step timing. This page is intended for monitoring long DIA-NN pipeline runs while they are still executing.

## UpSet item levels

The UpSet plot can aggregate N-terminal evidence at different item levels:

- `precursor`: protein + start + stripped peptide + N-terminal status + charge.
- `nterm_peptide`: protein + start + stripped peptide + N-terminal status.
- `nterm_site`: protein + start + N-terminal status.
- `protein`: protein only.

For `nterm_peptide` and `nterm_site`, the N-terminal modification/status is part of the item identity. If the same peptide or site is observed as both free and dimethyl, it becomes two UpSet rows instead of one row with a combined `nterm_status`.

## FDR basis

All UpSet item tables are filtered on the DIA-NN `Q.Value` column before aggregation.

- The active threshold is the selected `qvalue_threshold`.
- The displayed item score is the minimum DIA-NN `Q.Value` observed for that item.
- If no explicit threshold is selected, the server keeps all reported DIA-NN `Q.Value` rows.
