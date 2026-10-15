# DIA-NN CLI Reference & Workflow Documentation

This repository documents the DIA-NN command-line parameters used in a structured 4-step workflow:

1. Pre-search / candidate filtering
2. Predicted spectral library generation
3. Model fine-tuning
4. Final dataset quantification

This document consolidates relevant parameters from:
- Official DIA-NN documentation
- Common workflow practice
- Advanced tuning usage

---

# 1. Core Input / Output Parameters

## Data Input

--dir <path>\
Directory containing raw DIA files (.mzML, .raw).

--fasta <path>\
FASTA reference file. Can be specified multiple times.

--lib <path>\
Input spectral library.

--cont-quant-exclude <pattern>\
Exclude proteins matching prefix from quantification (e.g., cRAP-).

---

## Output

--out <path>\
Main output report file (.parquet or .tsv).

--out-lib <path>\
Export generated spectral library.

--matrices\
Export quantification matrices.

--export-quant\
Export fragment-level quantification data.

--qvalue <float>\
False discovery rate threshold (e.g., 0.01 for 1%).

--reanalyse\
Force re-analysis even if output files already exist.

---

# 2. Search Space & Digestion

--cut "<rules>"\
Protease cleavage rule (e.g., K*,R* for trypsin).

--missed-cleavages <int>\
Number of allowed missed cleavages.

--semi\
Enable semi-specific digestion.

--min-pep-len <int>\
Minimum peptide length.

--max-pep-len <int>\
Maximum peptide length.

--min-pr-mz <float>\
Minimum precursor m/z.

--max-pr-mz <float>\
Maximum precursor m/z.

--min-pr-charge <int>\
Minimum precursor charge.

--max-pr-charge <int>\
Maximum precursor charge.

--min-fr-mz <float>\
Minimum fragment m/z.

--max-fr-mz <float>\
Maximum fragment m/z.

--met-excision\
Remove initial methionine if biologically plausible.

---

# 3. Modifications

--unimod4\
Fixed carbamidomethylation (C).

--var-mods <int>\
Maximum variable modifications per peptide.

--var-mod "<UniMod:ID,mass,target>"\
Define variable modification.

Examples:

--var-mod UniMod:35,15.994915,M        # Methionine oxidation\
--var-mod UniMod:1,42.010565,n         # N-terminal acetylation\
--var-mod UniMod:36,28.031300,nK       # Dimethylation\

---

# 4. Mass Accuracy & Instrument Settings

--mass-acc <ppm>\
MS2 fragment mass tolerance (ppm).

--mass-acc-ms1 <ppm>\
MS1 precursor mass tolerance (ppm).

--high-acc\
Enable high-accuracy quantification strategy.

--direct-quant\
Legacy direct quantification mode.

---

# 5. Library Generation

--gen-spec-lib\
Generate spectral library during run.

--predictor\
Use deep learning predictor for in-silico library.

--fasta-search\
Perform FASTA-based digestion search.

---

# 6. Pre-Search / Candidate Filtering (Optional Step)

--pre-search\
Enable preliminary search.

--pre-filter\
Coarse candidate filtering before detailed scoring.

--time-corr-only\
Use chromatographic correlation only.

--extracted-ms1\
Include MS1 signal in candidate scoring.

--min-cal <int>\
Minimum calibration points.

--min-class <int>\
Minimum classification points.

--no-rt-window\
Disable RT window restriction.

--pre-select <int>\
Maximum candidate precursors passed to scoring.

--pre-select-force\
Force pre-selection even for small datasets.

---

# 7. Retention Time & Profiling

--rt-profiling\
Enable retention time profiling.

--im-window <float>\
Ion mobility extraction window.

--no-ifs-removal\
Disable interference filtering.

---

# 8. Model Fine-Tuning

--tune-lib <path>\
Input empirical library for tuning.

--tune-rt\
Fine-tune retention time prediction model.

--tune-im\
Fine-tune ion mobility prediction model.

--tune-fr\
Fine-tune fragmentation model.

--tune-lr <float>\
Learning rate override for tuning.

--tune-restrict-layers\
Restrict which layers are updated.

When using tuned models:

--tokens <file>\
Dictionary file from tuning.

--rt-model <file>\
Tuned RT model.

--im-model <file>\
Tuned IM model.

---

# 9. Recommended Workflow Order

1. Pre-search (optional)
2. Generate predicted library
3. Generate empirical tuning library
4. Tune RT/IM/fragment models
5. Final quantification with tuned models

---

# Notes

- Always keep digestion and modification settings consistent between steps.
- Avoid overly large variable modification space.
- Compare runs with and without tuned models to evaluate performance gain.
- Fix mass accuracy if reproducibility is required.

---

Official repository:
https://github.com/vdemichev/DiaNN
