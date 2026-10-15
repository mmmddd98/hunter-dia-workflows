# MAPLOT

MAPLOT creates quality-control and comparison plots from DIA-NN result files. It can load Parquet, TSV, and CSV reports, apply a configurable q-value filter, annotate peptide termini when the required FASTA is provided, and save figures for one search or for comparisons across searches.

## Install

From this directory, install the package and its declared dependencies:

```bash
python -m pip install -e .
```

## Generate a report

Provide an input report and an output directory. Use paths to your own data; no data files are bundled with this package.

```bash
python run.py /path/to/final_report.parquet --outdir ./plots
```

For the full command line options, run `python run.py --help`.

## Use the Python API

```python
from maplot import PlotProjectConfig, PlotProject

config = PlotProjectConfig.from_file(
    input_file="/path/to/final_report.parquet",
    output_dir="./plots",
)
project = PlotProject(config)
project.run()
```

`PlotProjectConfig` also provides experiment-based configuration for the standard DIA-NN output layout. The plotting API includes run counts, label efficiency, retention-time distributions, and UpSet summaries. See `src/maplot/plotters/` for the available plotters and `run.py --help` for supported options.

## Inputs and output

Input reports must contain the DIA-NN columns required by the selected plot. FASTA-based termini annotation requires a compatible protein FASTA. Generated plots and intermediate tables are written only to the output directory selected by the user.

Keep raw files, reports, FASTA databases, and generated outputs outside the source tree.
