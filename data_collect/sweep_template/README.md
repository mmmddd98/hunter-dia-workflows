# Resumable GPU sweep template

This template runs independent experiment jobs across configurable GPU slots. Use it for DIA-NN, Carafe2, and related sweeps where each job fits on one GPU.

## Properties

- configurable GPUs and concurrent jobs per GPU;
- default free-VRAM guard that reduces or disables workers on GPUs already in use;
- dynamic job distribution across free workers;
- one append-only log per job;
- live total and per-slot progress bars;
- atomic completion markers tied to the exact command configuration;
- per-job locks against duplicate execution;
- optional cross-process GPU-slot locks so separately launched sweeps share a fixed capacity;
- fail-fast behavior while already-running jobs finish cleanly;
- safe resume after interruption or crash;
- dry-run support.

## Manifest

Copy `example_manifest.json` and generate one entry per independent run. Commands are argv arrays and are executed without a shell. Multiple commands may be listed for training followed by evaluation.

`completion_files` should contain the final validated artifacts, not intermediate checkpoints. A job is skipped only when all completion files and a matching completion marker exist.

The optional `progress_regex` must capture the current step or epoch in group 1. It may capture the total in group 2; alternatively set `progress_total` in the manifest.

## Start

```bash
cd /path/to/carafe2-workspace

python3 ma/data_collect/sweep_template/parallel_gpu_sweep.py \
  --manifest ma/data_collect/sweep_template/example_manifest.json \
  --gpus 0,1,2 \
  --jobs-per-gpu 2 \
  --shared-slot-root /tmp/my_study_gpu_slots \
  --dry-run
```

Remove `--dry-run` to execute. Repeating the same command resumes the sweep and skips valid completed jobs.

Before starting workers, the runner queries `nvidia-smi` and caps each GPU to
`floor((free MiB - 256 MiB) / 3200 MiB)` workers. These defaults match the
observed Carafe2 RT peak. Override the estimate with
`--gpu-memory-per-job-mib`, or set exact capacities with, for example,
`--jobs-per-gpu-map 0:2,1:1,2:0`.

## Recommended workflow for a new study

1. Generate deterministic train/test inputs.
2. Generate a manifest containing every seed/configuration combination.
3. Point `completion_files` at final metrics or another trustworthy terminal artifact.
4. Test the manifest with `--dry-run`.
5. Start the sweep in a terminal to see live bars.
6. Re-run the same command after a crash or planned interruption.

Use the same shared lock policy across related launchers when two studies must not oversubscribe the GPUs. If studies may run together, divide the available slots explicitly instead of giving every study the full GPU capacity.

When two manifests use the same `--shared-slot-root`, `GPU N · slot M` is a global cross-process capacity. Waiting workers display `waiting for shared GPU slot`; locks are automatically released on normal exit or process termination.
