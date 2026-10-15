from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import sys
from functools import lru_cache
from pathlib import Path
from collections.abc import Iterable

import pandas as pd
import plotly.graph_objects as go

from .figures import PLOTLY_TEMPLATE, SELECTION_PLOT_HEIGHT, empty_figure, property_fraction_bar, property_violin
from .set_analysis import normalize_evidence_filters


MOTIF_LABELS = ["P4", "P3", "P2", "P1", "P1'", "P2'", "P3'", "P4'"]
MOTIF_WINDOW = {"upstream": 4, "downstream": 4, "anchor": "P1|P1'", "start_position": "1-based"}
AA_ALPHABET = list("ACDEFGHIKLMNPQRSTVWY")
MIN_LOGO_WINDOWS = 3
PYDAGLOGO_SRC = Path(__file__).resolve().parents[1] / "external" / "pydaglogo" / "src"


def load_fasta_sequences(fasta_path: Path | None) -> dict[str, str]:
    if fasta_path is None or not fasta_path.exists():
        return {}
    return _load_fasta_sequences_cached(str(fasta_path))


@lru_cache(maxsize=8)
def _load_fasta_sequences_cached(fasta_path: str) -> dict[str, str]:
    sequences: dict[str, str] = {}
    current_ids: list[str] = []
    chunks: list[str] = []
    with Path(fasta_path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                _store_sequence(sequences, current_ids, chunks)
                current_ids = fasta_identifier_aliases(line[1:])
                chunks = []
            else:
                chunks.append(line)
    _store_sequence(sequences, current_ids, chunks)
    return sequences


def _store_sequence(sequences: dict[str, str], ids: list[str], chunks: list[str]) -> None:
    if not ids or not chunks:
        return
    sequence = re.sub(r"[^A-Z]", "", "".join(chunks).upper())
    if not sequence:
        return
    for identifier in ids:
        sequences.setdefault(identifier, sequence)


def fasta_identifier_aliases(header: str) -> list[str]:
    token = header.split()[0]
    aliases = [token]
    parts = token.split("|")
    if len(parts) >= 2:
        aliases.append(parts[1])
    if len(parts) >= 3:
        aliases.append(parts[2])
    return list(dict.fromkeys(alias for alias in aliases if alias))


def site_table_from_evidence(evidence_df: pd.DataFrame, sequences: dict[str, str]) -> pd.DataFrame:
    if evidence_df.empty or not sequences:
        return pd.DataFrame(columns=["protein_id", "start_position", "protein_length", "relative_nterm_position", "p1_residue", "p1_prime_residue", "motif_window"])
    required = {"protein_id", "peptide_start"}
    if not required.issubset(evidence_df.columns):
        return pd.DataFrame(columns=["protein_id", "start_position"])

    rows = []
    work = evidence_df[["protein_id", "peptide_start"]].copy()
    work["protein_id"] = work["protein_id"].fillna("").astype(str).map(normalize_protein_id)
    work["start_position"] = pd.to_numeric(work["peptide_start"], errors="coerce")
    work = work.dropna(subset=["protein_id", "start_position"]).drop_duplicates(["protein_id", "start_position"])
    for row in work.itertuples(index=False):
        protein_id = str(row.protein_id)
        start = int(row.start_position)
        sequence = resolve_sequence(protein_id, sequences)
        if not sequence or start < 1 or start > len(sequence):
            continue
        p1 = sequence[start - 2] if start >= 2 else None
        p1_prime = sequence[start - 1] if start <= len(sequence) else None
        window = sequence[start - 5 : start + 3] if start >= 5 and start + 3 <= len(sequence) else None
        if window is not None and (len(window) != 8 or any(aa not in AA_ALPHABET for aa in window)):
            window = None
        rows.append(
            {
                "protein_id": protein_id,
                "start_position": start,
                "protein_length": len(sequence),
                "relative_nterm_position": start / len(sequence),
                "p1_residue": p1 if p1 in AA_ALPHABET else None,
                "p1_prime_residue": p1_prime if p1_prime in AA_ALPHABET else None,
                "motif_window": window,
            }
        )
    return pd.DataFrame(rows)


def normalize_protein_id(value: str) -> str:
    text = str(value or "").strip()
    text = re.sub(r" .*$", "", text)
    text = re.sub(r";.*$", "", text)
    return text


def resolve_sequence(protein_id: str, sequences: dict[str, str]) -> str | None:
    if protein_id in sequences:
        return sequences[protein_id]
    parts = protein_id.split("|")
    for part in parts:
        if part in sequences:
            return sequences[part]
    return None


def position_motif_figure(data: pd.DataFrame, plot_id: str) -> go.Figure:
    if data.empty:
        return empty_figure("No valid N-terminal site rows for this plot.")
    if plot_id == "relative_nterm_position":
        return property_violin(data, "relative_nterm_position", "Relative N-terminus position")
    if plot_id == "p1_distribution":
        return property_fraction_bar(data, "p1_residue", "P1 residue distribution", "P1 residue")
    if plot_id == "p1_prime_distribution":
        return property_fraction_bar(data, "p1_prime_residue", "P1' residue distribution", "P1' residue")
    if plot_id == "sequence_logo":
        mode = data.attrs.get("logo_mode", "selected")
        background_windows = data.attrs.get("background_windows")
        return create_sequence_logo(data["motif_window"].dropna().astype(str).tolist(), background_windows, mode=mode)
    return empty_figure("Unknown position or motif plot.")


def create_sequence_logo(foreground_windows: list[str], background_windows: list[str] | None = None, mode: str = "selected") -> go.Figure:
    foreground = valid_windows(foreground_windows)
    background = valid_windows(background_windows or [])
    if len(foreground) < MIN_LOGO_WINDOWS:
        return empty_figure(f"Need at least {MIN_LOGO_WINDOWS} valid full motif windows; found {len(foreground)}.")
    if mode != "selected" and len(background) < MIN_LOGO_WINDOWS:
        return empty_figure(f"Need at least {MIN_LOGO_WINDOWS} valid background motif windows; found {len(background)}.")

    try:
        if str(PYDAGLOGO_SRC) not in sys.path:
            sys.path.insert(0, str(PYDAGLOGO_SRC))
        from pydaglogo import build_background_model_from_samples, make_daglogo, plot_logo
    except Exception as exc:
        return empty_figure(f"pydaglogo is not available: {exc}")

    try:
        if mode == "selected":
            matrix = residue_frequency_matrix(foreground)
            ax, _matrix = plot_logo(matrix, labels=MOTIF_LABELS, y_label="Residue frequency", pvalue_cutoff=None)
        else:
            background_model = build_background_model_from_samples(
                [windows_to_matrix(background)],
                test_type="fisher",
                source_background=mode,
                model="any",
                target_position="any",
            )
            _result, ax, _matrix = make_daglogo(
                foreground,
                background_model=background_model,
                input_mode="windows",
                metric="difference",
                plot_mode="signed",
                labels=MOTIF_LABELS,
                pvalue_cutoff=None,
                analysis_p_adjust_method="none",
            )
        ax.axvline(3.5, color="#111111", linewidth=1.15)
        title = "Cleavage-site sequence logo" if mode == "selected" else "Cleavage-site sequence logo: selected vs background"
        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.tick_params(axis="both", labelsize=10)
        return matplotlib_axis_to_plotly(ax, title)
    except Exception as exc:
        return empty_figure(f"Could not create sequence logo: {exc}")


def valid_windows(windows: list[str]) -> list[str]:
    return [window for window in (str(w).upper() for w in windows) if len(window) == 8 and all(aa in AA_ALPHABET for aa in window)]


def residue_frequency_matrix(windows: list[str]) -> pd.DataFrame:
    rows = []
    for position in range(8):
        counts = {aa: 0 for aa in AA_ALPHABET}
        for window in windows:
            counts[window[position]] += 1
        denom = max(len(windows), 1)
        rows.append({aa: counts[aa] / denom for aa in AA_ALPHABET})
    return pd.DataFrame(rows, columns=AA_ALPHABET)


def windows_to_matrix(windows: list[str]) -> pd.DataFrame:
    return pd.DataFrame([{f"V{idx + 1}": aa for idx, aa in enumerate(window)} for window in windows])


def matplotlib_axis_to_plotly(ax, title: str) -> go.Figure:
    figure = ax.figure
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=240, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(figure)
    except Exception:
        pass
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    fig = go.Figure()
    fig.add_layout_image(
        dict(
            source=f"data:image/png;base64,{encoded}",
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            sizex=1,
            sizey=1,
            sizing="contain",
            layer="above",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=title,
        height=max(SELECTION_PLOT_HEIGHT, 380),
        margin=dict(l=30, r=30, t=68, b=32),
        xaxis={"visible": False, "range": [0, 1]},
        yaxis={"visible": False, "range": [0, 1]},
    )
    return fig


def cached_background_windows(
    *,
    cache_root: Path,
    dataset_id: str,
    item_level: str,
    evidence_filter: str | Iterable[str],
    background_mode: str,
    searches: list[str],
    evidence_df: pd.DataFrame,
    sequences: dict[str, str],
) -> list[str]:
    evidence_filter_key = "+".join(normalize_evidence_filters(evidence_filter))
    evidence_filter_key = evidence_filter_key or "unknown"
    key_payload = {
        "dataset_id": dataset_id,
        "item_level": item_level,
        "evidence_filter": evidence_filter_key,
        "background_mode": background_mode,
        "searches": sorted(searches),
        "search_hash": hashlib.sha256("|".join(sorted(searches)).encode("utf-8")).hexdigest(),
        "motif_window": MOTIF_WINDOW,
    }
    digest = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    safe_parts = [safe_path_part(part) for part in (dataset_id, item_level, evidence_filter_key, background_mode)]
    cache_dir = cache_root / "motif_backgrounds" / safe_parts[0] / safe_parts[1] / safe_parts[2] / safe_parts[3]
    windows_path = cache_dir / f"{digest}.windows.csv"
    meta_path = cache_dir / f"{digest}.json"
    if windows_path.exists():
        cached = pd.read_csv(windows_path)
        if "motif_window" in cached.columns:
            return valid_windows(cached["motif_window"].dropna().astype(str).tolist())
    sites = site_table_from_evidence(evidence_df, sequences)
    windows = valid_windows(sites.get("motif_window", pd.Series(dtype=str)).dropna().astype(str).tolist())
    cache_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"motif_window": windows}).to_csv(windows_path, index=False)
    meta_path.write_text(json.dumps({**key_payload, "n_windows": len(windows)}, indent=2, sort_keys=True), encoding="utf-8")
    return windows


def safe_path_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_") or "unknown"
