# HUNTER DIA Workflows

Workflows and analysis tools developed for a master's thesis on peptide-model
adaptation in HUNTER DIA proteomics.

## Components

| Component | Purpose |
| --- | --- |
| [DIA-NN pipeline](diaNN_pipline/README.md) | Predicted-library generation, integrated model fine-tuning and searches of held-out files |
| [RT dataset tools](data_collect/README.md) | Retention-time data preparation, alignment and model-evaluation utilities |
| [Report processing](diann_visu/post_run_reports/README.md) | DIA-NN report processing, N-terminal annotation and comparison plots |
| [MAPLOT server](diann_visu/maplotsserver/README.md) | Interactive exploration and quality control of search results |
| [Entrapment generation](entrapment/README.md) | Shuffled protein-sequence generation for entrapment analyses |

## Getting started

```bash
git clone https://github.com/mmmddd98/hunter-dia-workflows.git
cd hunter-dia-workflows
```

Follow the relevant component README for installation and configuration.
Dependencies and external software requirements differ between components.

Four benchmark configurations retain the thesis training/test assignments.
Measurement data, FASTA databases, trained models and generated results must
be supplied separately.

## Thesis

See [HUNTER DIA model adaptation](https://github.com/mmmddd98/hunter-dia-model-adaptation)
for the thesis, findings and related software repositories.

## License

[MIT](LICENSE).
