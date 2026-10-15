# MAPLOT report integration

The DIA-NN pipeline can generate a MAPLOT report after Step 5. The report summarizes quality-control and comparison plots from the resulting DIA-NN reports.

## Configure

Set the report options in `config/defaults.cfg` or override them for one run:

```bash
ENABLE_MAPLOT_REPORT=1
MAPLOT_PYTHON_PATH=python
MAPLOT_REPORT_SCRIPT=/path/to/ma_code/diann_visu/post_run_reports/scripts/run_maplot_report.py
MAPLOT_ORGANISM=human
MAPLOT_REUSE_CACHE=0
```

The default output root and other data locations come from the selected local machine configuration. Keep these settings out of version control.

## Run

```bash
bash run_all.sh --ds <dataset> --expname <experiment>
```

To disable automatic report generation for one run:

```bash
bash run_all.sh --ds <dataset> --expname <experiment> --no-maplot-report
```

Or set `ENABLE_MAPLOT_REPORT=0`. Reports are written below the selected experiment output directory. The standalone commands and plot options are documented in [`post_run_reports/README.md`](../../diann_visu/post_run_reports/README.md).
