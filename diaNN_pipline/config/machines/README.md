# Machine configuration

Machine configuration files contain local paths and executable locations. Keep them out of version control.

1. Copy `template.cfg` to a local file such as `local.cfg`.
2. Set the DIA-NN executable, raw-data root, FASTA root, dataset root, output root, temporary directory, and Python interpreters.
3. Select the configuration by setting `DIANN_MACHINE` to its filename without `.cfg`.

```bash
cp diaNN_pipline/config/machines/template.cfg diaNN_pipline/config/machines/local.cfg
export DIANN_MACHINE=local
bash diaNN_pipline/bin/check_machine_config.sh
```

Dataset YAML files use paths relative to `DATASET_ROOT`; FASTA names are resolved relative to `FASTABASE`. Use the same dataset definition across machines by configuring these roots appropriately.
