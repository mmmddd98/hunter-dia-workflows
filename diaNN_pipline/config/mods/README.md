# Modification presets

Choose a preset with the `--mods` option or the `MOD_CONFIG` setting:

```bash
bash run_all.sh --ds <dataset> --expname <experiment> --mods fixed_lysin_dimethyl
```

The `.cfg` files in this directory define fixed and variable modifications for DIA-NN. Their names describe the intended configuration; they do not contain dataset-specific paths or results. Keep the selected modification and digestion settings consistent across library generation, tuning, and quantification when comparing runs.

The `baseline_dimethyl255` preset declares a mass for UniMod:255 that may differ from the mass known to DIA-NN. Confirm the chemistry and DIA-NN warning behavior before using it.
