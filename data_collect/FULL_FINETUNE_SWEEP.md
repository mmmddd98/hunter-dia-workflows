# Full multi-organism RT fine-tuning strategy sweep

This experiment keeps the complete C. elegans + TripleMix + Human training pool fixed and varies only optimization and transfer-learning policy.

## Design

- Two independent Bayesian W&B sweeps: one DIA-NN and one Carafe2.
- 120 sampled configurations per model on seeds 2024 and 2025.
- Four fixed controls per model on seeds 2024 and 2025.
- Twelve validation-selected finalists per model confirmed on seeds 2026–2033.
- One winner per model selected using validation only and then evaluated on the untouched final test.
- Total: `(120 + 4) × 2 seeds × 2 models + 12 × 8 additional seeds × 2 models = 688` fine-tuning runs.

The optimized validation objective is:

`0.4 × macro global MAE + 0.3 × macro Heavy-Dimethyl MAE + 0.3 × macro N-term Heavy-Dimethyl MAE`.

The search varies MAE/L1, mathematically matched Huber, MSE, exact-stratum Heavy-Dimethyl weights, learning rate, weight decay, and standard head/partial/full/discriminative fine-tuning policies. Oversampling stays fixed at 1.0.

## Prepared data

The default prepared directory is:

`$FINETUNE_SWEEP_OUTPUT_ROOT/full_multiorganism_finetune_sweep_v1/prepared`

The split is globally sequence-disjoint across all three source organisms:

- training: 53,886 rows / 47,470 unique sequences
- validation: 6,803 rows / 5,993 unique sequences
- final test: 6,889 rows / 6,022 unique sequences

## Start broad sweeps

```bash
nohup ./data_collect/start_full_finetune_strategy_sweeps.sh \
  > "$FINETUNE_SWEEP_OUTPUT_ROOT/full_multiorganism_finetune_sweep_v1/start.log" 2>&1 &
```

Defaults use CUDA device 0 for DIA-NN and device 1 for Carafe2. Override with `DIANN_CUDA_DEVICE`, `CARAFE2_CUDA_DEVICE`, `SWEEP_HOURS`, or `SWEEP_CONFIGS_PER_MODEL`.

After both broad sweeps finish, the script automatically creates CSV summaries, PNG/PDF/SVG figures, an HTML report, and the twelve finalist configurations per model.

## Confirm finalists and run final held-out test

```bash
nohup ./data_collect/confirm_full_finetune_finalists.sh \
  > "$FINETUNE_SWEEP_OUTPUT_ROOT/full_multiorganism_finetune_sweep_v1/confirmation.log" 2>&1 &
```

The held-out test is evaluated only after the winner has been chosen from multi-seed validation results.

## Status

```bash
./data_collect/status_full_finetune_sweeps.sh
```

## Thesis-ready outputs

Outputs are written under `.../full_multiorganism_finetune_sweep_v1/analysis`:

- `sweep_runs.csv`
- `sweep_configuration_summary.csv`
- `sweep_report.html`
- `figure_manifest.json`
- `fig1_configuration_ranking.{png,pdf,svg}`
- `fig2_global_nterm_pareto.{png,pdf,svg}`
- `fig3_policy_loss_comparison.{png,pdf,svg}`
- `fig4_heavy_weight_response.{png,pdf,svg}`
- `fig5_multiseed_confirmation.{png,pdf,svg}`
- `fig6_final_heldout_test.{png,pdf,svg}`

PNG files are exported at 450 dpi; PDF and SVG remain vector-editable for the thesis.
