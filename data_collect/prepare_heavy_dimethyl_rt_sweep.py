#!/usr/bin/env python3
"""Prepare the validation-only Heavy-Dimethyl RT sweeps using the standard runner.

This generator only writes manifests, input links and pre-registered study
metadata.  It never reads or references a held-out test partition.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from heavy_dimethyl_sweep_common import (
    BATCH_SIZE, EMBEDDING_LEARNING_RATES, EMBEDDING_PROFILES, EPOCHS, LEARNING_RATES,
    COMPACT_BALANCED_OVERSAMPLE_PROFILES, COMPACT_BALANCED_WEIGHT_PROFILES,
    COMPACT_DOSE_OVERSAMPLE_PROFILES, COMPACT_DOSE_WEIGHT_PROFILES, COMPACT_LOW_LR_LEARNING_RATES,
    LOW_LR_DOSE_OVERSAMPLE_PROFILES, LOW_LR_DOSE_WEIGHT_PROFILES, LOW_LR_LEARNING_RATES,
    MODES, OVERSAMPLE_PROFILES, SEEDS, WARMUP_EPOCHS, WEIGHT_PROFILES, lr_label,
    oversampling_budget, profile_text, target_scale_and_huber_delta, training_counts,
)

ROOT = Path(__file__).resolve().parents[2]
DIANN = Path(os.environ.get("DIANN_FINETUNE_ROOT", ROOT.parent / "diann-fintune"))
DIANN_MODEL = Path(os.environ.get("DIANN_MODEL_DIR", ROOT.parent / "diann_sc/DIA-NN-2.5.0/diann-2.5.0/models"))
CARAFE2 = Path(os.environ.get("CARAFE2_ROOT", ROOT / "carafe2-semi"))
CARAFE2_TRAINER = CARAFE2 / "src/main/resources/py/v2/ai.py"
DATASET_DEFAULT = ROOT / "trainingdata/finalrt_4source_trainfit15_validation10_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("diann", "carafe2"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--arm", choices=("main", "low_lr", "low_lr_dose", "compact_low_lr_dose", "compact_low_lr_dose_balanced", "embedding", "smoke"), default="main")
    parser.add_argument("--diann-python", default=os.environ.get("DIANN_PYTHON", sys.executable))
    fallback = Path("python")
    parser.add_argument("--carafe-python", default=os.environ.get("CARAFE2_PYTHON", str(fallback if fallback.is_file() else Path(sys.executable))))
    return parser.parse_args()


def require_inputs(data: Path) -> dict[str, dict[str, Path]]:
    inputs = {}
    for mode in MODES:
        base = data / mode
        values = {
            "train": base / "train/diann_rt_input/rt_train_data.tsv",
            "validation": base / "validation/diann_rt_input/rt_validation_data.tsv",
            "partition": base / "validation", "alignment": base / "rt_alignment.json",
        }
        missing = [item for item in (*values.values(), values["partition"] / "rt_observations.tsv") if not item.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {mode} study inputs: {missing}")
        if any("test" in str(value).lower() for value in values.values()):
            raise ValueError("Study input unexpectedly references test data")
        inputs[mode] = values
    return inputs


def diann_job(python: str, values: dict[str, Path], run: Path, *, seed: int, lr: float,
              loss: str, huber_delta: float, weights: tuple[float, float, float],
              oversampling: tuple[float, float, float], policy: str = "all") -> dict:
    train = [python, str(DIANN / "train_carafe_rt.py"), "--train-split", str(values["train"]),
             "--validation-split", str(values["validation"]), "--model-dir", str(DIANN_MODEL),
             "--output-dir", str(run), "--device", "cuda", "--epochs", str(EPOCHS),
             "--warmup-epochs", str(WARMUP_EPOCHS), "--batch-size", str(BATCH_SIZE),
             "--loss", loss, "--huber-delta", f"{huber_delta:.12g}", "--weight-decay", "0",
             "--heavy-nterm-weight", str(weights[0]), "--heavy-lysine-weight", str(weights[1]),
             "--heavy-both-weight", str(weights[2]), "--heavy-nterm-oversample", str(oversampling[0]),
             "--heavy-lysine-oversample", str(oversampling[1]), "--heavy-both-oversample", str(oversampling[2]),
             "--seed", str(seed), "--trainable", policy, "--learning-rate", str(lr), "--no-plots"]
    evaluation = [python, str(ROOT / "ma/data_collect/evaluate_diann_trainfit.py"), "--run-dir", str(run),
                  "--partition-dir", str(values["partition"]), "--alignment", str(values["alignment"]),
                  "--output", str(run / "validation_metrics.json"), "--device", "cuda"]
    return {"workdir": str(DIANN), "commands": [train, evaluation]}


def carafe_job(python: str, input_dir: Path, values: dict[str, Path], run: Path, *, seed: int,
               lr: float, loss: str, huber_delta: float, weights: tuple[float, float, float],
               oversampling: tuple[float, float, float]) -> dict:
    train = [python, str(CARAFE2_TRAINER), "--in_dir", str(input_dir), "--out_dir", str(run),
             "--device", "gpu", "--tf_type", "rt", "--seed", str(seed), "--verbose", "1", "--use_best_model",
             "--rt_validation_file", str(values["validation"]), "--rt_validation_disjoint_by", "peptidoform",
             "--rt_epoch", str(EPOCHS), "--rt_warmup_epoch", str(WARMUP_EPOCHS), "--rt_batch_size", str(BATCH_SIZE),
             "--no_adjust_rt_batch_size", "--rt_loss", "l1" if loss == "mae" else "huber",
             "--rt_huber_delta", f"{huber_delta:.12g}", "--rt_weight_decay", "0",
             "--rt_heavy_nterm_oversample", str(oversampling[0]), "--rt_heavy_lysine_oversample", str(oversampling[1]),
             "--rt_heavy_both_oversample", str(oversampling[2]), "--rt_heavy_nterm_weight", str(weights[0]),
             "--rt_heavy_lysine_weight", str(weights[1]), "--rt_heavy_both_weight", str(weights[2]),
             "--rt_trainable", "all", "--rt_lr", str(lr)]
    evaluation = [python, str(ROOT / "ma/data_collect/evaluate_carafe2_trainfit.py"), "--run-dir", str(run),
                  "--partition-dir", str(values["partition"]), "--alignment", str(values["alignment"]),
                  "--output", str(run / "validation_metrics.json"), "--device", "gpu", "--python", python]
    return {"workdir": str(CARAFE2), "commands": [train, evaluation]}


def add_job(jobs: list[dict], job: dict, job_id: str, metadata: dict) -> None:
    evaluation = job["commands"][-1]
    completion = Path(evaluation[evaluation.index("--output") + 1])
    # DIA-NN emits ``epoch=7/40``; Carafe2 emits ``Epoch=7, ...``.
    progress_regex = r"epoch=(\d+)/(\d+)" if metadata["model"] == "diann" else r"Epoch=(\d+)"
    job.update({"id": job_id, "completion_files": [str(completion)],
                "progress_regex": progress_regex, "progress_total": EPOCHS, "metadata": metadata})
    jobs.append(job)


def design_rows(family: str, arm: str):
    if arm == "embedding":
        if family != "diann":
            raise ValueError("The embedding diagnostic is DIA-NN only")
        for mode in MODES:
            for profile, weights in EMBEDDING_PROFILES.items():
                for lr in EMBEDDING_LEARNING_RATES:
                    for seed in SEEDS:
                        yield "embedding", mode, profile, weights, (1., 1., 1.), lr, "mae", seed, "embeddings"
        return
    # A genuinely small end-to-end smoke design: control, weighting and
    # oversampling for each model, plus the DIA-NN embedding diagnostic.
    if arm == "smoke":
        mode, lr, seed = "chronologer_hi", LEARNING_RATES[family][1], SEEDS[0]
        yield "weighting", mode, "W0", WEIGHT_PROFILES["W0"], (1., 1., 1.), lr, "mae", seed, "all"
        yield "weighting", mode, "W10", WEIGHT_PROFILES["W10"], (1., 1., 1.), lr, "mae", seed, "all"
        yield "oversampling", mode, "O5", (1., 1., 1.), OVERSAMPLE_PROFILES["O5"], lr, "mae", seed, "all"
        if family == "diann":
            yield "embedding", mode, "W10", WEIGHT_PROFILES["W10"], (1., 1., 1.), EMBEDDING_LEARNING_RATES[1], "mae", seed, "embeddings"
        return
    if arm == "low_lr":
        # Controls stay duplicated by arm because the matched-control analysis
        # treats weighting and oversampling as separate interventions.  W5/W6
        # and O3/O4 are the moderate and stronger profiles that were most
        # informative in the main study; all other variables are frozen.
        for mode in MODES:
            for profile in ("W0", "W5", "W6"):
                for lr in LOW_LR_LEARNING_RATES:
                    for seed in SEEDS:
                        yield "weighting", mode, profile, WEIGHT_PROFILES[profile], (1., 1., 1.), lr, "mae", seed, "all"
            for profile in ("O0", "O3", "O4"):
                for lr in LOW_LR_LEARNING_RATES:
                    for seed in SEEDS:
                        yield "oversampling", mode, profile, (1., 1., 1.), OVERSAMPLE_PROFILES[profile], lr, "mae", seed, "all"
        return
    if arm == "low_lr_dose":
        # Full 1x--10x heavy-both dose response across the already selected
        # low-LR bracket.  Weighting and oversampling are deliberately kept in
        # separate arms with their own controls.
        for mode in MODES:
            for profile in LOW_LR_DOSE_WEIGHT_PROFILES:
                for lr in LOW_LR_LEARNING_RATES:
                    for seed in SEEDS:
                        yield "weighting", mode, profile, WEIGHT_PROFILES[profile], (1., 1., 1.), lr, "mae", seed, "all"
            for profile in LOW_LR_DOSE_OVERSAMPLE_PROFILES:
                for lr in LOW_LR_LEARNING_RATES:
                    for seed in SEEDS:
                        yield "oversampling", mode, profile, (1., 1., 1.), OVERSAMPLE_PROFILES[profile], lr, "mae", seed, "all"
        return
    if arm == "compact_low_lr_dose":
        # One C0 run is an identical 1x/1x baseline for both mechanisms.
        # Reusing it removes a mathematically redundant training run without
        # weakening the seed-matched comparison.
        for mode in MODES:
            for lr in COMPACT_LOW_LR_LEARNING_RATES:
                for seed in SEEDS:
                    yield "control", mode, "C0", (1., 1., 1.), (1., 1., 1.), lr, "mae", seed, "all"
                    for profile in COMPACT_DOSE_WEIGHT_PROFILES:
                        yield "weighting", mode, profile, WEIGHT_PROFILES[profile], (1., 1., 1.), lr, "mae", seed, "all"
                    for profile in COMPACT_DOSE_OVERSAMPLE_PROFILES:
                        yield "oversampling", mode, profile, (1., 1., 1.), OVERSAMPLE_PROFILES[profile], lr, "mae", seed, "all"
        return
    if arm == "compact_low_lr_dose_balanced":
        # The balanced companion adds N-term and lysine at half of the
        # heavy-both dose.  1/1/2 already exists in the pure-heavy-both series,
        # so only 2/2/4, 4/4/8 and 5/5/10 require additional jobs.
        for mode in MODES:
            for lr in COMPACT_LOW_LR_LEARNING_RATES:
                for seed in SEEDS:
                    yield "control", mode, "C0", (1., 1., 1.), (1., 1., 1.), lr, "mae", seed, "all"
                    for profile in COMPACT_DOSE_WEIGHT_PROFILES:
                        yield "weighting", mode, profile, WEIGHT_PROFILES[profile], (1., 1., 1.), lr, "mae", seed, "all"
                    for profile in COMPACT_BALANCED_WEIGHT_PROFILES:
                        yield "weighting", mode, profile, WEIGHT_PROFILES[profile], (1., 1., 1.), lr, "mae", seed, "all"
                    for profile in COMPACT_DOSE_OVERSAMPLE_PROFILES:
                        yield "oversampling", mode, profile, (1., 1., 1.), OVERSAMPLE_PROFILES[profile], lr, "mae", seed, "all"
                    for profile in COMPACT_BALANCED_OVERSAMPLE_PROFILES:
                        yield "oversampling", mode, profile, (1., 1., 1.), OVERSAMPLE_PROFILES[profile], lr, "mae", seed, "all"
        return
    weight_items = list(WEIGHT_PROFILES.items())
    oversample_items = list(OVERSAMPLE_PROFILES.items())
    losses = ("mae", "huber")
    for mode in MODES:
        for profile, weights in weight_items:
            for lr in LEARNING_RATES[family]:
                for loss in losses:
                    for seed in SEEDS:
                        yield "weighting", mode, profile, weights, (1., 1., 1.), lr, loss, seed, "all"
        for profile, oversampling in oversample_items:
            for lr in LEARNING_RATES[family]:
                for seed in SEEDS:
                    yield "oversampling", mode, profile, (1., 1., 1.), oversampling, lr, "mae", seed, "all"


def validate_design(jobs: list[dict], family: str, arm: str) -> None:
    ids = [job["id"] for job in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate heavy-study manifest job IDs")
    if arm in ("main", "low_lr", "low_lr_dose", "compact_low_lr_dose", "compact_low_lr_dose_balanced"):
        expected = {
            "main": 720,
            "low_lr": 2 * 6 * len(LOW_LR_LEARNING_RATES) * len(SEEDS),
            "low_lr_dose": 2 * 12 * len(LOW_LR_LEARNING_RATES) * len(SEEDS),
            "compact_low_lr_dose": 2 * len(COMPACT_LOW_LR_LEARNING_RATES) * len(SEEDS) * (1 + len(COMPACT_DOSE_WEIGHT_PROFILES) + len(COMPACT_DOSE_OVERSAMPLE_PROFILES)),
            "compact_low_lr_dose_balanced": 2 * len(COMPACT_LOW_LR_LEARNING_RATES) * len(SEEDS) * (1 + len(COMPACT_DOSE_WEIGHT_PROFILES) + len(COMPACT_BALANCED_WEIGHT_PROFILES) + len(COMPACT_DOSE_OVERSAMPLE_PROFILES) + len(COMPACT_BALANCED_OVERSAMPLE_PROFILES)),
        }[arm]
        if len(jobs) != expected:
            raise ValueError(f"Expected {expected} main jobs, found {len(jobs)}")
        controls = {(j["metadata"]["arm"], j["metadata"]["target"], j["metadata"]["loss"], j["metadata"]["lr"], j["metadata"]["seed"])
                    for j in jobs if j["metadata"]["profile"] in ("W0", "O0")}
        shared_controls = {(j["metadata"]["target"], j["metadata"]["loss"], j["metadata"]["lr"], j["metadata"]["seed"])
                           for j in jobs if j["metadata"]["profile"] == "C0"}
        for job in jobs:
            meta = job["metadata"]
            if meta["arm"] == "weighting" and meta["oversampling"] != "1/1/1": raise ValueError("Weighting arm oversamples")
            if meta["arm"] == "oversampling" and meta["weights"] != "1/1/1": raise ValueError("Oversampling arm reweights")
            if meta["profile"] == "C0" and (meta["weights"] != "1/1/1" or meta["oversampling"] != "1/1/1"):
                raise ValueError("Shared control is not an unmodified baseline")
            if arm in ("compact_low_lr_dose", "compact_low_lr_dose_balanced"):
                if meta["profile"] != "C0":
                    key = (meta["target"], meta["loss"], meta["lr"], meta["seed"])
                    if key not in shared_controls: raise ValueError(f"Missing shared control for {job['id']}")
            elif meta["profile"] not in ("W0", "O0"):
                key = (meta["arm"], meta["target"], meta["loss"], meta["lr"], meta["seed"])
                if key not in controls: raise ValueError(f"Missing matched control for {job['id']}")
            if arm in ("low_lr", "low_lr_dose") and (meta["loss"] != "mae" or meta["lr"] not in LOW_LR_LEARNING_RATES):
                raise ValueError(f"Low-LR job outside the frozen MAE/LR design: {job['id']}")
            if arm == "compact_low_lr_dose" and (meta["loss"] != "mae" or meta["lr"] not in COMPACT_LOW_LR_LEARNING_RATES):
                raise ValueError(f"Compact low-LR job outside the frozen MAE/LR design: {job['id']}")
            if arm == "compact_low_lr_dose_balanced" and (meta["loss"] != "mae" or meta["lr"] not in COMPACT_LOW_LR_LEARNING_RATES):
                raise ValueError(f"Balanced compact low-LR job outside the frozen MAE/LR design: {job['id']}")
    if arm == "embedding" and len(jobs) != 96: raise ValueError(f"Expected 96 embedding jobs, found {len(jobs)}")


def write_readme(output: Path, family: str, arm: str, scales: dict, budgets: dict) -> None:
    label = "DIA-NN" if family == "diann" else "Carafe2"
    if arm == "low_lr":
        design = f"""This follow-up deliberately tests only low learning rates: {', '.join(f'`{lr:.2g}`' for lr in LOW_LR_LEARNING_RATES)}. The previous 720-run DIA-NN study already covered `2e-4`--`6e-4`; `2e-4` and `3e-4` are retained as overlap points.

It fixes MAE, policy `all`, 40 epochs, 10 warm-up epochs and batch size 512. Weighting tests `W0`/`W5`/`W6` = `1/1/1`, `1/1/2`, `1/1/4`; oversampling tests `O0`/`O3`/`O4` = `1/1/1`, `1/1/2`, `1/1/4`. With two target spaces and three seeds, this is **{2 * 6 * len(LOW_LR_LEARNING_RATES) * len(SEEDS)} jobs**.
"""
    elif arm == "low_lr_dose":
        design = f"""This validation-only dose-response follow-up uses low learning rates {', '.join(f'`{lr:.2g}`' for lr in LOW_LR_LEARNING_RATES)}; `2e-4` and `3e-4` overlap with the previous large DIA-NN sweep.

It fixes MAE, policy `all`, 40 epochs, 10 warm-up epochs and batch size 512. For the doubly dimethylated stratum only, loss weighting tests `W0`/`W5`/`W6`/`W7`/`W12`/`W13` = `1x/2x/4x/6x/8x/10x`; oversampling tests `O0`/`O3`/`O4`/`O6`/`O7`/`O8` = `1x/2x/4x/6x/8x/10x`. Weighting leaves row counts unchanged; oversampling uses the native trainer behavior and can raise the update budget. With two target spaces and three seeds, this is **504 jobs** per model.
"""
    elif arm == "compact_low_lr_dose":
        design = f"""This compact validation-only dose-response sweep uses {', '.join(f'`{lr:.2g}`' for lr in COMPACT_LOW_LR_LEARNING_RATES)}. The final two values, `2e-4` and `3e-4`, anchor the new experiment to the previous 720-run DIA-NN grid.

It fixes MAE, policy `all`, 40 epochs, 10 warm-up epochs and batch size 512. For the doubly dimethylated stratum only, both loss weighting and oversampling test `1x/2x/4x/8x/10x`. The `1x` baseline is a single shared `C0` run because `1x` loss weighting plus `1x` oversampling is exactly the same training configuration. Thus each target/LR/seed has one control, four loss-weighting runs and four oversampling runs: **270 jobs per model**. Weighting leaves row counts unchanged; oversampling uses native trainer behavior and can raise the update budget.
"""
    elif arm == "compact_low_lr_dose_balanced":
        design = f"""This compact validation-only dose-response sweep uses {', '.join(f'`{lr:.2g}`' for lr in COMPACT_LOW_LR_LEARNING_RATES)}. `2e-4` and `3e-4` anchor it to the previous 720-run DIA-NN grid.

It includes the pure Heavy-both series `1/1/1`, `1/1/2`, `1/1/4`, `1/1/8`, `1/1/10` and a balanced companion series in which N-term and lysine each receive half the Heavy-both factor: `2/2/4`, `4/4/8`, `5/5/10`. The already-present `1/1/2` is the balanced 2x Heavy-both point, so it is not duplicated. The same profiles are tested separately as loss weighting and as oversampling; `C0=1/1/1` is shared by both mechanisms. This is **450 jobs per model**: 30 controls, 210 loss-weighting and 210 oversampling runs. Weighting leaves row counts unchanged; oversampling uses native trainer behavior and can raise the update budget.
"""
    else:
        design = f"The primary main arm has 576 loss-weighting and 144 oversampling jobs ({'96 separate DIA-NN embedding jobs' if arm == 'embedding' else '720 jobs'})."
    research_questions = """
## Forschungsfragen und erwartete Aussagen

Dieses Experiment soll nicht einfach den kleinsten MAE-Wert suchen, sondern die folgenden voneinander trennbaren Fragen beantworten:

1. **Lernrate:** Liegt für die Heavy-Dimethyl-Anpassung ein stabilerer bzw. besserer Bereich unterhalb der bisherigen Suche? Die Werte `2e-4` und `3e-4` verbinden die neue Reihe mit dem großen DIA-NN-Sweep; für Carafe2 ist `3e-4` zugleich ein direkter Anker zum bisherigen Bereich.
2. **Mechanismus:** Hilft es mehr, den Fehler dieser Peptide im Loss stärker zu gewichten, oder sie im Training häufiger zu zeigen? Lossgewichtung verändert bei gleicher Datenmenge die Gradientenbeiträge; Oversampling verändert die Stichprobenhäufigkeit und mit der nativen Trainer-Logik auch das Update-Budget.
3. **Dosisantwort für Heavy-both:** Verbessert ein höherer Faktor (`1x` bis `10x`) die Fehler der doppelt dimethylierten Peptide tatsächlich, und gibt es einen Punkt ab dem der Nutzen sättigt oder kippt?
4. **Beitrag der Einzelklassen:** Reicht eine reine Heavy-both-Intervention (`1/1/x`) aus, oder hilft es, die nur N-terminal bzw. nur Lysin-modifizierten Peptide moderat mitzunehmen (`x/2`, mit mindestens `1x`)? Die Paare `1/1/4` vs. `2/2/4`, `1/1/8` vs. `4/4/8` und `1/1/10` vs. `5/5/10` isolieren genau diese Frage.
5. **Robustheit statt Spezialgewinn:** Ein Kandidat ist nur überzeugend, wenn der Gewinn über die drei Seeds und beide Targets repliziert und nicht mit unvertretbaren Verschlechterungen bei Overall, Non-heavy, Semi-tryptic oder einer einzelnen Quelle/Organismusgruppe bezahlt wird.

### Erwartete Entscheidung nach dem Sweep

Für jede Modellfamilie wird die **kleinste** Dosis bevorzugt, die den Heavy-both-MAE reproduzierbar verbessert und die Guardrails für Overall, Non-heavy, Semi-tryptic und Worst-source einhält. Die Organismus- und Stratum-Auswertung prüft zusätzlich, ob ein scheinbarer Gesamtgewinn nur von einem Organismus oder einer Peptidklasse getragen wird. Ein hoher Oversampling-Faktor wird daher nicht automatisch als mechanistisch besser interpretiert: Er ist nur dann vorzuziehen, wenn sein zusätzlicher Nutzen den zugleich erhöhten Trainingsumfang rechtfertigt.
""" if arm == "compact_low_lr_dose_balanced" else ""
    text = f"""# {label} Heavy-Dimethyl RT sweep ({arm})

Validation-only manifest generated by `prepare_heavy_dimethyl_rt_sweep.py` using the existing trainer wrappers, evaluators and `parallel_gpu_sweep.py` resume/fingerprint runner. The held-out test partition is intentionally absent from every command.

## Fixed contract

Targets: `chronologer_hi`, `anchor_irt`; epochs/warm-up/batch: {EPOCHS}/{WARMUP_EPOCHS}/{BATCH_SIZE}; weight decay: 0; seeds: {', '.join(map(str, SEEDS))}.

{design}

{research_questions}

## Huber scale conversion

Both trainers apply Huber in `rt_norm` coordinates. A pre-registered 1.5-minute transition is converted by affine fit on the fixed training data:

| Target | minutes / target unit | Huber delta (target units) |
|---|---:|---:|
""" + "\n".join(f"| `{mode}` | {value['raw_minutes_per_target_unit']:.6f} | {value['huber_delta_target_units']:.8f} |" for mode, value in scales.items()) + f"""

## Oversampling step-budget note

Existing trainer semantics are preserved. Oversampling duplicates rows after the fixed split, so it increases effective rows and normally optimizer/scheduler/warm-up steps. This is recorded per job in `metadata.training_budget`; the low-LR sweep therefore tests the practical combined effect of oversampling and its native update budget, not a step-budget-normalized causal sampling effect.

The pre-launch data/argument/fingerprint sanity checks and full frozen design are in `sweep_manifest.json`.
"""
    output.joinpath("README.md").write_text(text, encoding="utf-8")
    output.joinpath("training_budget_by_mode.json").write_text(json.dumps(budgets, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args(); output = args.output_dir.resolve(); data = args.dataset_dir.resolve()
    if output.exists(): raise FileExistsError(f"Refusing to overwrite existing study: {output}")
    if args.family == "diann" and not (DIANN / "train_carafe_rt.py").is_file(): raise FileNotFoundError("DIA-NN trainer unavailable")
    if args.family == "carafe2" and (not CARAFE2_TRAINER.is_file() or not Path(args.carafe_python).is_file()): raise FileNotFoundError("Carafe2 trainer/interpreter unavailable")
    inputs = require_inputs(data); output.mkdir(parents=True)
    if args.family == "carafe2":
        for mode, values in inputs.items():
            target = output / "prepared_inputs" / mode; target.mkdir(parents=True)
            target.joinpath("rt_train_data.tsv").symlink_to(values["train"])
    jobs: list[dict] = []; scales = {}; budgets = {}
    for mode, values in inputs.items():
        scales[mode] = target_scale_and_huber_delta(values["train"])
        counts = training_counts(values["train"]); budgets[mode] = {"base_counts": counts}
        for arm, _, profile, weights, oversampling, lr, loss, seed, policy in design_rows(args.family, args.arm):
            if _ != mode: continue
            run = output / "runs" / arm / mode / loss / profile / lr_label(lr) / f"seed_{seed}"
            delta = scales[mode]["huber_delta_target_units"]
            job = (diann_job(args.diann_python, values, run, seed=seed, lr=lr, loss=loss, huber_delta=delta, weights=weights, oversampling=oversampling, policy=policy)
                   if args.family == "diann" else carafe_job(str(Path(args.carafe_python).resolve()), output / "prepared_inputs" / mode, values, run, seed=seed, lr=lr, loss=loss, huber_delta=delta, weights=weights, oversampling=oversampling))
            budget = oversampling_budget(counts, oversampling); budgets[mode][profile] = budget
            metadata = {"model": args.family, "arm": arm, "target": mode, "policy": policy, "profile": profile,
                        "loss": loss, "lr": lr, "seed": seed, "weights": profile_text(weights),
                        "oversampling": profile_text(oversampling), "huber": scales[mode], "training_budget": budget,
                        "validation_only": True, "test_partition_referenced": False}
            add_job(jobs, job, f"{arm}/{mode}/{loss}/{profile}/{lr_label(lr)}/seed_{seed}", metadata)
    validate_design(jobs, args.family, args.arm)
    study = f"{args.family}_heavy_dimethyl_rt_{args.arm}_v1"
    manifest = {"study": study, "dataset": str(data), "validation_only": True, "test_partition": "closed",
                "design_arm": args.arm, "huber_conversion": scales, "jobs": jobs}
    output.joinpath("sweep_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_readme(output, args.family, args.arm, scales, budgets)
    print(json.dumps({"study": study, "jobs": len(jobs), "manifest": str(output / 'sweep_manifest.json'), "validation_only": True}, indent=2))


if __name__ == "__main__": main()
