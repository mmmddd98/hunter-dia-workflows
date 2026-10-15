from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Literal

import polars as pl


EntityLevel = Literal["peptide", "peptidoform", "nterm", "protein"]
UpSetItemLevel = Literal["precursor", "nterm_peptide", "nterm_site", "protein"]
EvidenceFilter = Literal["relevant", "dimethyl", "acetyl", "free", "all", "exclude_free"]

AA_ALPHABET = tuple("ACDEFGHIKLMNPQRSTVWY")
ENTITY_LEVEL_LABELS: dict[EntityLevel, str] = {
    "peptide": "Stripped peptide",
    "peptidoform": "Modified sequence",
    "nterm": "N-terminal peptidoform",
    "protein": "Protein group",
}
ENTITY_LEVEL_DESCRIPTIONS: dict[EntityLevel, str] = {
    "peptide": "unique stripped peptide sequence",
    "peptidoform": "unique DIA-NN Modified.Sequence including modifications",
    "nterm": "unique nterm_modif_peptide: N-terminal modification class plus stripped peptide",
    "protein": "unique protein group identifier",
}
ENTITY_COLUMNS: dict[EntityLevel, tuple[str, ...]] = {
    "peptide": ("Stripped.Sequence", "peptide"),
    "peptidoform": ("Modified.Sequence",),
    "nterm": ("nterm_modif_peptide",),
    "protein": ("Protein.Group", "Protein.Ids", "protein"),
}
UPSET_ITEM_LEVEL_LABELS: dict[UpSetItemLevel, str] = {
    "precursor": "Precursor",
    "nterm_peptide": "N-terminal peptide",
    "nterm_site": "N-terminal site",
    "protein": "Protein",
}
UPSET_ITEM_LEVEL_DESCRIPTIONS: dict[UpSetItemLevel, str] = {
    "precursor": "protein + start + stripped peptide + N-term status + charge; charge states stay separate.",
    "nterm_peptide": "protein + start + stripped peptide + N-term status; charge and non-N-terminal variable mods are aggregated.",
    "nterm_site": "protein + start + N-term status; peptide length, missed cleavages, charge, and variable mods are aggregated.",
    "protein": "protein only; all selected N-terminal evidence for the protein is aggregated.",
}
EVIDENCE_FILTER_LABELS: dict[EvidenceFilter, str] = {
    "relevant": "Relevant only: Dimethyl + Acetyl",
    "dimethyl": "Dimethyl only",
    "acetyl": "Acetyl only",
    "free": "Free NH3 only",
    "all": "All observed",
    "exclude_free": "Exclude free NH3",
}
UPSET_LEVEL_TO_ID_PARTS: dict[UpSetItemLevel, tuple[str, ...]] = {
    "precursor": ("protein_id", "peptide_start", "stripped_sequence", "nterm_status", "charge"),
    "nterm_peptide": ("protein_id", "peptide_start", "stripped_sequence", "nterm_status"),
    "nterm_site": ("protein_id", "peptide_start", "nterm_status"),
    "protein": ("protein_id",),
}
NORMALIZED_EVIDENCE_COLUMNS = (
    "protein_id",
    "peptide_start",
    "modified_sequence",
    "stripped_sequence",
    "charge",
    "precursor_id",
    "nterm_status",
)
EVIDENCE_COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "protein_id": ("protein_id", "protein", "Protein.Ids", "Protein.Group", "Protein.Names"),
    "peptide_start": ("peptide_start", "start_position", "Start.Position", "Protein.Start"),
    "modified_sequence": ("modified_sequence", "Modified.Sequence", "nterm_modif_peptide"),
    "stripped_sequence": ("stripped_sequence", "Stripped.Sequence", "peptide"),
    "charge": ("charge", "Precursor.Charge", "Charge"),
    "precursor_id": ("precursor_id", "Precursor.Id", "PrecursorID", "Precursor"),
    "nterm_status": ("nterm_status",),
}


@dataclass(frozen=True)
class IntersectionSelection:
    """Exact set membership selected from a multi-search comparison."""

    key: str
    present_searches: tuple[str, ...]


def normalize_upset_item_level(value: str | None) -> UpSetItemLevel:
    aliases: dict[str, UpSetItemLevel] = {
        "peptidoform": "nterm_peptide",
        "nterm": "nterm_site",
        "peptide": "nterm_peptide",
        "modified_nterm_peptide": "nterm_peptide",
        "unique_nterminus": "nterm_site",
        "raw_peptidoform": "nterm_peptide",
        "raw_peptide": "nterm_peptide",
        "raw_protein": "protein",
    }
    if value in UPSET_LEVEL_TO_ID_PARTS:
        return value  # type: ignore[return-value]
    return aliases.get(value or "", "nterm_site")


def normalize_evidence_filter(value: str | None) -> EvidenceFilter:
    if value in EVIDENCE_FILTER_LABELS:
        return value  # type: ignore[return-value]
    return "relevant"


def normalize_evidence_filters(value: str | Iterable[str] | None) -> tuple[EvidenceFilter, ...]:
    if value is None:
        return ("relevant",)
    if isinstance(value, str):
        return (normalize_evidence_filter(value),)
    selected = {normalize_evidence_filter(item) for item in value}
    if not selected:
        return ("relevant",)
    if "all" in selected:
        return ("all",)
    if "free" in selected and "exclude_free" in selected:
        return ("all",)
    if "exclude_free" in selected:
        return ("exclude_free",)
    normalized: list[EvidenceFilter] = []
    if "relevant" in selected:
        normalized.append("relevant")
    elif "dimethyl" in selected or "acetyl" in selected:
        if "dimethyl" in selected:
            normalized.append("dimethyl")
        if "acetyl" in selected:
            normalized.append("acetyl")
    if "free" in selected:
        normalized.append("free")
    return tuple(normalized or ["relevant"])


def evidence_filter_label(value: str | Iterable[str] | None) -> str:
    filters = normalize_evidence_filters(value)
    short_labels = {
        "dimethyl": "Dimethyl",
        "acetyl": "Acetyl",
        "free": "Free NH3",
        "relevant": "Dimethyl + Acetyl",
        "all": "All observed",
        "exclude_free": "Exclude free NH3",
    }
    if "all" in filters:
        return EVIDENCE_FILTER_LABELS["all"]
    labels = [short_labels.get(filt, EVIDENCE_FILTER_LABELS[filt]) for filt in filters]
    return labels[0] if len(labels) == 1 else " + ".join(labels)


def prepare_upset_items(
    tables: dict[str, pl.DataFrame],
    searches: Iterable[str],
    item_level: str | None,
    evidence_filter: str | Iterable[str] | None,
    *,
    include_entrapment: bool = False,
) -> tuple[pl.DataFrame, list[str]]:
    """Build one boolean UpSet row per filtered N-terminal item.

    The row-level evidence filter is applied before item identifiers are
    aggregated. This keeps unique N-terminus and protein levels evidence-aware.
    """

    level = normalize_upset_item_level(item_level)
    filt = normalize_evidence_filters(evidence_filter)
    per_search: dict[str, pl.DataFrame] = {}
    for search in searches:
        df = tables.get(search)
        if df is None or df.is_empty():
            continue
        normalized = normalize_nterm_evidence_table(df, include_entrapment=include_entrapment)
        filtered = apply_nterm_evidence_filter(normalized, filt)
        with_item = add_upset_item_id(filtered, level)
        if with_item.is_empty():
            continue
        per_search[search] = with_item

    set_cols = [search for search in searches if search in per_search]
    if not set_cols:
        return empty_upset_item_frame(), []

    all_items = sorted(
        set().union(*(set(df.get_column("item_id").to_list()) for df in per_search.values()))
    )
    members_by_search = {search: set(df.get_column("item_id").to_list()) for search, df in per_search.items()}
    bool_rows = []
    for item_id in all_items:
        row = {"item_id": item_id, "entity": item_id}
        for search in set_cols:
            row[search] = item_id in members_by_search[search]
        bool_rows.append(row)
    bool_df = pl.DataFrame(bool_rows)

    metadata = upset_item_metadata(per_search, level)
    if metadata.is_empty():
        return bool_df, set_cols
    return bool_df.join(metadata, on="item_id", how="left"), set_cols


def prepare_upset_evidence_rows(
    tables: dict[str, pl.DataFrame],
    searches: Iterable[str],
    item_level: str | None,
    evidence_filter: str | Iterable[str] | None,
    *,
    include_entrapment: bool = False,
) -> pl.DataFrame:
    level = normalize_upset_item_level(item_level)
    filt = normalize_evidence_filters(evidence_filter)
    rows = []
    for search in searches:
        df = tables.get(search)
        if df is None or df.is_empty():
            continue
        normalized = normalize_nterm_evidence_table(df, include_entrapment=include_entrapment)
        filtered = apply_nterm_evidence_filter(normalized, filt)
        with_item = add_upset_item_id(filtered, level)
        if with_item.is_empty():
            continue
        rows.append(with_item.with_columns(pl.lit(search).alias("search")))
    if not rows:
        return empty_upset_evidence_frame()
    return pl.concat(rows, how="diagonal_relaxed")


def normalize_nterm_evidence_table(df: pl.DataFrame, *, include_entrapment: bool = False) -> pl.DataFrame:
    work = df
    if not include_entrapment and "is_entrapment" in work.columns:
        work = work.filter(~pl.col("is_entrapment").cast(pl.Boolean, strict=False).fill_null(False))

    exprs: list[pl.Expr] = []
    for normalized in NORMALIZED_EVIDENCE_COLUMNS:
        col = first_existing_column(work, EVIDENCE_COLUMN_CANDIDATES[normalized])
        if col is not None:
            exprs.append(pl.col(col).cast(pl.Utf8, strict=False).alias(normalized))
        else:
            exprs.append(pl.lit(None, dtype=pl.Utf8).alias(normalized))

    for optional in (
        "protein_organism",
        "nterm_modif",
        "specificity",
        "id_origin",
        "Q.Value",
        "Global.Q.Value",
        "Lib.Q.Value",
        "Peptidoform.Q.Value",
        "Global.Peptidoform.Q.Value",
        "Lib.Peptidoform.Q.Value",
        "PG.Q.Value",
        "Global.PG.Q.Value",
        "Lib.PG.Q.Value",
        "Protein.Q.Value",
        "PEP",
        "RT",
        "Predicted.RT",
        "Best.Fr.Mz.Delta",
        "Ms1.Apex.Mz.Delta",
        "Precursor.Normalised",
        "is_entrapment",
        "is_diann_decoy",
        "Protein.Sites",
    ):
        if optional in work.columns:
            exprs.append(pl.col(optional).alias(optional))

    out = work.select(exprs)
    if "nterm_modif" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.Utf8).alias("nterm_modif"))
    if "Protein.Sites" in out.columns:
        out = out.drop("Protein.Sites")
    out = out.with_columns(
        pl.col("protein_id")
        .str.replace(r" .*$", "")
        .str.replace(r";.*$", "")
        .alias("protein_id")
    )
    if "stripped_sequence" in out.columns and "modified_sequence" in out.columns:
        out = out.with_columns(
            pl.when(pl.col("stripped_sequence").is_null() | (pl.col("stripped_sequence") == ""))
            .then(pl.col("modified_sequence").map_elements(strip_modifications, return_dtype=pl.Utf8))
            .otherwise(pl.col("stripped_sequence"))
            .alias("stripped_sequence")
        )
    out = out.with_columns(
        pl.when(
            (pl.col("modified_sequence").is_null() | (pl.col("modified_sequence") == ""))
            & pl.col("nterm_modif").is_not_null()
            & pl.col("stripped_sequence").is_not_null()
        )
        .then(pl.concat_str([pl.col("nterm_modif"), pl.col("stripped_sequence")], separator="_"))
        .otherwise(pl.col("modified_sequence"))
        .alias("modified_sequence")
    )
    out = out.with_columns(
        pl.struct(["nterm_status", "nterm_modif", "modified_sequence"])
        .map_elements(status_from_evidence_struct, return_dtype=pl.Utf8)
        .alias("nterm_status")
    )
    out = out.with_columns(
        pl.when(pl.col("precursor_id").is_null() | (pl.col("precursor_id") == ""))
        .then(
            pl.concat_str(
                [
                    pl.col("protein_id").fill_null(""),
                    pl.col("peptide_start").fill_null(""),
                    pl.col("modified_sequence").fill_null(""),
                    pl.col("charge").fill_null(""),
                ],
                separator="|",
            )
        )
        .otherwise(pl.col("precursor_id"))
        .alias("precursor_id")
    )
    return out


def raw_with_termini_evidence(raw_df: pl.DataFrame, termini_df: pl.DataFrame, *, include_entrapment: bool = False) -> pl.DataFrame:
    """Attach termini positions to raw precursor rows without losing precursor detail."""

    raw_input = raw_df
    if "Protein.Ids" in raw_input.columns:
        raw_input = (
            raw_input.with_columns(pl.col("Protein.Ids").cast(pl.Utf8).str.split(";").alias("Protein.Ids"))
            .explode("Protein.Ids")
            .with_columns(pl.col("Protein.Ids").str.strip_chars().alias("Protein.Ids"))
        )
    raw = normalize_nterm_evidence_table(raw_input, include_entrapment=include_entrapment)
    termini = normalize_nterm_evidence_table(termini_df, include_entrapment=include_entrapment)
    if raw.is_empty() or termini.is_empty():
        return raw

    join_cols = ["protein_id", "stripped_sequence", "nterm_status"]
    term_cols = [
        col
        for col in [
            "protein_id",
            "stripped_sequence",
            "nterm_status",
            "peptide_start",
            "protein_organism",
            "specificity",
            "nterm_modif",
            "id_origin",
            "is_entrapment",
        ]
        if col in termini.columns
    ]
    term_meta = termini.select(term_cols).drop_nulls(["protein_id", "stripped_sequence", "peptide_start"]).unique()
    if term_meta.is_empty():
        return raw

    joined_metadata = (
        "peptide_start",
        "protein_organism",
        "specificity",
        "nterm_modif",
        "id_origin",
        "is_entrapment",
    )
    rename = {col: f"{col}_termini" for col in joined_metadata if col in term_meta.columns}
    term_meta = term_meta.rename(rename)
    joined = raw.join(term_meta, on=join_cols, how="left")
    fill_exprs = []
    for col in joined_metadata:
        term_col = f"{col}_termini"
        if term_col in joined.columns:
            if col not in joined.columns:
                joined = joined.with_columns(pl.lit(None).alias(col))
            fill_exprs.append(
                pl.when(pl.col(term_col).is_not_null())
                .then(pl.col(term_col))
                .otherwise(pl.col(col))
                .alias(col)
            )
    if fill_exprs:
        joined = joined.with_columns(fill_exprs).drop([col for col in joined.columns if col.endswith("_termini")])
    return joined


def apply_nterm_evidence_filter(df: pl.DataFrame, evidence_filter: EvidenceFilter | Iterable[EvidenceFilter]) -> pl.DataFrame:
    filters = normalize_evidence_filters(evidence_filter)
    if df.is_empty() or "all" in filters:
        return df
    masks: list[pl.Expr] = []
    for evidence_filter in filters:
        if evidence_filter == "relevant":
            masks.append(pl.col("nterm_status").is_in(["dimethyl", "acetyl"]))
        elif evidence_filter == "dimethyl":
            masks.append(pl.col("nterm_status") == "dimethyl")
        elif evidence_filter == "acetyl":
            masks.append(pl.col("nterm_status") == "acetyl")
        elif evidence_filter == "free":
            masks.append(pl.col("nterm_status") == "free")
        elif evidence_filter == "exclude_free":
            masks.append(pl.col("nterm_status") != "free")
    if not masks:
        return df
    mask = masks[0]
    for expr in masks[1:]:
        mask = mask | expr
    return df.filter(mask)


def add_upset_item_id(df: pl.DataFrame, item_level: UpSetItemLevel) -> pl.DataFrame:
    if df.is_empty():
        return df.with_columns(pl.lit(None, dtype=pl.Utf8).alias("item_id")).filter(pl.col("item_id").is_not_null())
    parts = UPSET_LEVEL_TO_ID_PARTS[item_level]
    required = [
        pl.col(part).is_not_null() & (pl.col(part).cast(pl.Utf8, strict=False).str.strip_chars() != "")
        for part in parts
    ]
    valid = required[0]
    for expr in required[1:]:
        valid = valid & expr
    return df.filter(valid).with_columns(
        pl.concat_str([pl.col(part).cast(pl.Utf8) for part in parts], separator="|").alias("item_id")
    )


def upset_item_metadata(per_search: dict[str, pl.DataFrame], item_level: UpSetItemLevel | None = None) -> pl.DataFrame:
    if not per_search:
        return empty_upset_item_frame()
    rows = []
    for search, df in per_search.items():
        rows.append(df.with_columns(pl.lit(search).alias("search")))
    long_df = pl.concat(rows, how="diagonal_relaxed")
    if long_df.is_empty():
        return empty_upset_item_frame()

    exprs: list[pl.Expr] = [
        text_expr("protein_id").drop_nulls().first().alias("protein_id"),
        text_expr("protein_id").drop_nulls().first().alias("protein"),
        text_expr("peptide_start").drop_nulls().first().alias("peptide_start"),
        text_expr("modified_sequence").drop_nulls().first().alias("modified_sequence"),
        text_expr("stripped_sequence").drop_nulls().first().alias("stripped_sequence"),
        text_expr("charge").drop_nulls().first().alias("charge"),
        text_expr("charge").drop_nulls().unique().str.join(";").alias("charge_states"),
        text_expr("precursor_id").drop_nulls().first().alias("precursor_id"),
        text_expr("nterm_status").drop_nulls().unique().str.join(";").alias("nterm_status"),
        text_expr("search").drop_nulls().unique().str.join(";").alias("observed_searches"),
        pl.len().alias("evidence_count"),
    ]
    if "stripped_sequence" in long_df.columns:
        exprs.append(pl.col("stripped_sequence").cast(pl.Utf8).str.len_chars().median().alias("peptide_length"))
    if "Q.Value" in long_df.columns:
        exprs.append(pl.col("Q.Value").cast(pl.Float64, strict=False).min().alias("score"))
    elif "PEP" in long_df.columns:
        exprs.append(pl.col("PEP").cast(pl.Float64, strict=False).min().alias("score"))
    if "Precursor.Normalised" in long_df.columns:
        intensity = pl.col("Precursor.Normalised").cast(pl.Float64, strict=False)
        exprs.append(intensity.sum().alias("summed_intensity"))
        exprs.append(intensity.sum().alias("intensity"))
    for candidate in ("RT", "Predicted.RT", "Best.Fr.Mz.Delta", "Ms1.Apex.Mz.Delta"):
        if candidate in long_df.columns:
            exprs.append(pl.col(candidate).cast(pl.Float64, strict=False).median().alias(safe_column_name(candidate)))
    for candidate in ("protein_organism", "nterm_modif", "specificity", "id_origin"):
        if candidate in long_df.columns:
            exprs.append(text_expr(candidate).drop_nulls().first().alias(candidate))
    if "is_entrapment" in long_df.columns:
        exprs.append(pl.col("is_entrapment").cast(pl.Boolean, strict=False).max().alias("is_entrapment"))

    metadata = long_df.group_by("item_id").agg(exprs)
    if "modified_sequence" in metadata.columns:
        metadata = metadata.with_columns(
            pl.col("modified_sequence").map_elements(strip_modifications, return_dtype=pl.Utf8).alias("sequence")
        )
    if "RT" in metadata.columns and "Predicted_RT" in metadata.columns:
        metadata = metadata.with_columns((pl.col("RT") - pl.col("Predicted_RT")).abs().alias("Abs_RT_Delta"))
    if "Best_Fr_Mz_Delta" in metadata.columns:
        metadata = metadata.with_columns(pl.col("Best_Fr_Mz_Delta").abs().alias("Abs_Best_Fr_Mz_Delta"))
    if "Ms1_Apex_Mz_Delta" in metadata.columns:
        metadata = metadata.with_columns(pl.col("Ms1_Apex_Mz_Delta").abs().alias("Abs_Ms1_Apex_Mz_Delta"))
    if "stripped_sequence" in metadata.columns:
        metadata = metadata.with_columns(
            pl.col("stripped_sequence").map_elements(gravy, return_dtype=pl.Float64).alias("gravy")
        )
    return metadata


def status_from_evidence_struct(values: dict[str, object]) -> str:
    existing = normalize_nterm_status(values.get("nterm_status"))
    if existing != "unknown":
        return existing
    for value in (values.get("nterm_modif"), values.get("modified_sequence")):
        status = infer_nterm_status(value)
        if status != "unknown":
            return status
    return "unknown"


def normalize_nterm_status(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null", "unknown"}:
        return "unknown"
    if "dimethyl" in text or "unimod:36" in text or "unimod:510" in text or "28.031" in text or "34.063" in text or "36.075" in text:
        return "dimethyl"
    if "acetyl" in text or "unimod:1" in text or "42.010" in text:
        return "acetyl"
    if text in {"n", "free", "free nh3", "free_nh3", "unmodified", "nh3"} or "free nh3" in text:
        return "free"
    if text in {"other"}:
        return "other"
    return "other"


def infer_nterm_status(value: object) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip().lower()
    if not text or text in {"nan", "none", "null"}:
        return "unknown"
    if "dimethyl" in text or "unimod:36" in text or "unimod:510" in text or "28.031" in text or "34.063" in text or "36.075" in text:
        return "dimethyl"
    if "acetyl" in text or "unimod:1" in text or "42.010" in text:
        return "acetyl"
    if text in {"n", "free", "free nh3", "free_nh3", "unmodified", "nh3"}:
        return "free"
    if not text.startswith("("):
        return "free"
    return "other"


def entity_sets(tables: dict[str, pl.DataFrame], entity_level: EntityLevel) -> dict[str, set[str]]:
    """Return unique entity identifiers per search.

    The function accepts raw DIA-NN tables, adapted peptide matrices, or
    termini-annotated tables. It picks the first suitable identifier column for
    the requested entity level.
    """

    sets: dict[str, set[str]] = {}
    for search, df in tables.items():
        expr = entity_identifier_expr(df, entity_level)
        if expr is None:
            sets[search] = set()
            continue
        sets[search] = set(
            df.select(expr.alias("entity"))
            .drop_nulls()
            .unique()
            .get_column("entity")
            .to_list()
        )
    return sets


def intersection_summary(
    tables: dict[str, pl.DataFrame],
    entity_level: EntityLevel,
    *,
    searches: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Build exact search-membership counts for one entity level."""

    sets = entity_sets(tables, entity_level)
    selected = tuple(searches or sorted(sets))
    if not selected:
        return empty_intersection_frame()

    membership: dict[str, tuple[str, ...]] = {}
    for search in selected:
        for entity in sets.get(search, set()):
            current = set(membership.get(entity, ()))
            current.add(search)
            membership[entity] = tuple(sorted(current))

    counts = Counter(membership.values())
    rows = []
    for present, count in counts.items():
        absent = tuple(search for search in selected if search not in present)
        rows.append(
            {
                "intersection_key": encode_intersection(present),
                "present_searches": " & ".join(present),
                "absent_searches": " & ".join(absent),
                "count": count,
                "n_searches": len(present),
            }
        )
    if not rows:
        return empty_intersection_frame()
    return pl.DataFrame(rows).sort(["count", "present_searches"], descending=[True, False])


def intersection_entities(
    tables: dict[str, pl.DataFrame],
    entity_level: EntityLevel,
    intersection_key: str,
    *,
    searches: Iterable[str] | None = None,
) -> list[str]:
    """Return entities that occur in exactly the selected searches."""

    sets = entity_sets(tables, entity_level)
    selected = tuple(searches or sorted(sets))
    present = set(decode_intersection(intersection_key))
    if not selected or not present:
        return []

    included = [sets.get(search, set()) for search in selected if search in present]
    if not included:
        return []
    entities = set.intersection(*included)
    for search in selected:
        if search not in present:
            entities -= sets.get(search, set())
    return sorted(entities)


def entity_feature_table(
    *,
    raw_tables: dict[str, pl.DataFrame] | None = None,
    adapted_tables: dict[str, pl.DataFrame] | None = None,
    termini_tables: dict[str, pl.DataFrame] | None = None,
    entity_level: EntityLevel,
    entities: Iterable[str] | None = None,
) -> pl.DataFrame:
    """Collect analyzable properties for entities across searches.

    The output is long-form: one row per search/entity where the entity is
    observed. It intentionally keeps only lightweight, broadly available
    features so the same table can power static reports and interactive Dash
    plots.
    """

    entity_filter = set(entities) if entities is not None else None
    rows: list[dict[str, object]] = []
    sources = tables_for_level(raw_tables, adapted_tables, termini_tables, entity_level)
    for search, df in sources.items():
        rows.extend(feature_rows_for_search(search, df, entity_level, entity_filter))
    if not rows:
        return empty_feature_frame()
    return pl.DataFrame(rows)


def add_intersection_labels(
    features: pl.DataFrame,
    source_tables: dict[str, pl.DataFrame],
    entity_level: EntityLevel,
) -> pl.DataFrame:
    """Attach exact membership labels to a feature table."""

    if features.is_empty() or "entity" not in features.columns:
        return features
    sets = entity_sets(source_tables, entity_level)
    labels = []
    for entity in features.get_column("entity").to_list():
        present = tuple(search for search in sorted(sets) if entity in sets[search])
        labels.append(encode_intersection(present))
    return features.with_columns(pl.Series("intersection_key", labels))


def encode_intersection(searches: Iterable[str]) -> str:
    return "||".join(sorted(searches))


def decode_intersection(key: str | None) -> tuple[str, ...]:
    if not key:
        return ()
    return tuple(part for part in key.split("||") if part)


def feature_rows_for_search(
    search: str,
    df: pl.DataFrame,
    entity_level: EntityLevel,
    entity_filter: set[str] | None,
) -> list[dict[str, object]]:
    entity_expr = entity_identifier_expr(df, entity_level)
    if entity_expr is None:
        return []

    work = df.with_columns(entity_expr.alias("entity")).drop_nulls("entity")
    if entity_filter is not None:
        work = work.filter(pl.col("entity").is_in(entity_filter))
    if work.is_empty():
        return []

    sequence_expr = sequence_expression(work, entity_level)
    base_exprs = [
        sequence_expr.drop_nulls().first().alias("sequence"),
        pl.col("entity").count().alias("rows"),
    ]
    for candidate in (
        "Q.Value",
        "PEP",
        "PG.Q.Value",
        "Protein.Q.Value",
        "RT",
        "Predicted.RT",
        "Best.Fr.Mz.Delta",
        "Ms1.Apex.Mz.Delta",
        "Lib.Q.Value",
        "Lib.Peptidoform.Q.Value",
        "Lib.PG.Q.Value",
        "Evidence",
        "Mass.Evidence",
        "Empirical.Quality",
        "Quantity.Quality",
        "Ms1.Profile.Corr",
        "Precursor.Charge",
    ):
        if candidate in work.columns:
            values = pl.col(candidate).cast(pl.Float64, strict=False)
            if candidate in {"Q.Value", "PEP", "PG.Q.Value", "Protein.Q.Value", "Lib.Q.Value", "Lib.Peptidoform.Q.Value", "Lib.PG.Q.Value"}:
                base_exprs.append(values.min().alias(candidate))
            else:
                base_exprs.append(values.median().alias(candidate))
    if "Precursor.Normalised" in work.columns:
        base_exprs.append(pl.col("Precursor.Normalised").cast(pl.Float64, strict=False).median().alias("abundance"))
    if "nterm_modif" in work.columns:
        base_exprs.append(pl.col("nterm_modif").cast(pl.Utf8).drop_nulls().first().alias("nterm_modif"))
    elif entity_level == "nterm":
        base_exprs.append(pl.col("entity").str.split_exact("_", 1).struct.field("field_0").alias("nterm_modif"))
    if "specificity" in work.columns:
        base_exprs.append(pl.col("specificity").cast(pl.Utf8).drop_nulls().first().alias("specificity"))
    if "id_origin" in work.columns:
        base_exprs.append(pl.col("id_origin").cast(pl.Utf8).drop_nulls().first().alias("id_origin"))
    if "is_entrapment" in work.columns:
        base_exprs.append(pl.col("is_entrapment").cast(pl.Boolean, strict=False).max().alias("is_entrapment"))
    if "protein" in work.columns:
        base_exprs.append(pl.col("protein").cast(pl.Utf8).drop_nulls().first().alias("protein"))
    elif "Protein.Ids" in work.columns:
        base_exprs.append(pl.col("Protein.Ids").cast(pl.Utf8).drop_nulls().first().alias("protein"))

    aggregated = work.group_by("entity").agg(base_exprs).with_columns(pl.lit(search).alias("search"))
    rows = []
    for row in aggregated.to_dicts():
        sequence = str(row.get("sequence") or "")
        row["length"] = len(sequence) if sequence else None
        row["nterm_residue"] = sequence[:1] or None
        row["cterm_residue"] = sequence[-1:] or None
        row["nterm_motif3"] = sequence[:3] or None
        row["cterm_motif3"] = sequence[-3:] or None
        row["gravy"] = gravy(sequence) if sequence else None
        if row.get("RT") is not None and row.get("Predicted.RT") is not None:
            row["Abs.RT.Delta"] = abs(float(row["RT"]) - float(row["Predicted.RT"]))
        if row.get("Best.Fr.Mz.Delta") is not None:
            row["Abs.Best.Fr.Mz.Delta"] = abs(float(row["Best.Fr.Mz.Delta"]))
        if row.get("Ms1.Apex.Mz.Delta") is not None:
            row["Abs.Ms1.Apex.Mz.Delta"] = abs(float(row["Ms1.Apex.Mz.Delta"]))
        for aa in AA_ALPHABET:
            row[f"aa_{aa}"] = sequence.count(aa) / len(sequence) if sequence else None
        rows.append(row)
    return rows


def tables_for_level(
    raw_tables: dict[str, pl.DataFrame] | None,
    adapted_tables: dict[str, pl.DataFrame] | None,
    termini_tables: dict[str, pl.DataFrame] | None,
    entity_level: EntityLevel,
) -> dict[str, pl.DataFrame]:
    if entity_level == "nterm":
        return termini_tables or adapted_tables or raw_tables or {}
    if entity_level == "peptide":
        return raw_tables or adapted_tables or termini_tables or {}
    if entity_level == "peptidoform":
        return raw_tables or {}
    return raw_tables or adapted_tables or termini_tables or {}


def sequence_expression(df: pl.DataFrame, entity_level: EntityLevel) -> pl.Expr:
    if entity_level == "nterm":
        if "peptide" in df.columns:
            return pl.col("peptide").cast(pl.Utf8)
        if "Stripped.Sequence" in df.columns:
            return pl.col("Stripped.Sequence").cast(pl.Utf8)
        return pl.col("entity").str.replace(r"^[^_]+_", "")
    if entity_level == "peptide":
        return pl.col("entity")
    if entity_level == "peptidoform":
        if "Stripped.Sequence" in df.columns:
            return pl.col("Stripped.Sequence").cast(pl.Utf8)
        return pl.col("entity").map_elements(strip_modifications, return_dtype=pl.Utf8)
    return pl.lit(None, dtype=pl.Utf8)


def first_existing_column(df: pl.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def entity_identifier_expr(df: pl.DataFrame, entity_level: EntityLevel) -> pl.Expr | None:
    col = first_existing_column(df, ENTITY_COLUMNS[entity_level])
    if col is not None:
        return pl.col(col).cast(pl.Utf8)
    if entity_level == "nterm" and {"Modified.Sequence", "Stripped.Sequence"}.issubset(df.columns):
        return (
            pl.col("Modified.Sequence")
            .cast(pl.Utf8)
            .map_elements(infer_nterm_modification, return_dtype=pl.Utf8)
            + pl.lit("_")
            + pl.col("Stripped.Sequence").cast(pl.Utf8)
        )
    return None


def infer_nterm_modification(modified_sequence: str | None) -> str:
    """Infer a compact N-term modification label from DIA-NN Modified.Sequence."""

    if not modified_sequence or not modified_sequence.startswith("("):
        return "n"
    match = re.match(r"^\(([^()]*)\)", modified_sequence)
    if not match:
        return "n"
    token = match.group(1)
    if any(marker in token for marker in ("UniMod:36", "UniMod:510", "28.031", "34.063", "36.075")):
        return "Dimethyl"
    if any(marker in token for marker in ("UniMod:737", "UniMod:738", "UniMod:2016", "UniMod:259")):
        return "TMT"
    if any(marker in token for marker in ("UniMod:1", "42.010")):
        return "Acetyl"
    if any(marker in token for marker in ("UniMod:26", "UniMod:27", "89.030")):
        return "2PCA"
    if "UniMod:35" in token or "15.994" in token:
        return "Oxidation"
    return token or "n"


def strip_modifications(sequence: str | None) -> str | None:
    if sequence is None:
        return None
    return re.sub(r"\([^()]*\)", "", sequence)


def text_expr(column: str) -> pl.Expr:
    return pl.col(column).map_elements(format_text_value, return_dtype=pl.Utf8)


def format_text_value(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return ";".join(str(item) for item in value if item is not None)
    return str(value)


def safe_column_name(column: str) -> str:
    return column.replace(".", "_")


def empty_intersection_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "intersection_key": pl.Utf8,
            "present_searches": pl.Utf8,
            "absent_searches": pl.Utf8,
            "count": pl.Int64,
            "n_searches": pl.Int64,
        }
    )


def empty_feature_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "entity": pl.Utf8,
            "sequence": pl.Utf8,
            "search": pl.Utf8,
            "rows": pl.Int64,
            "length": pl.Int64,
            "gravy": pl.Float64,
            "nterm_residue": pl.Utf8,
            "cterm_residue": pl.Utf8,
            "nterm_motif3": pl.Utf8,
            "cterm_motif3": pl.Utf8,
        }
    )


def empty_upset_item_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "item_id": pl.Utf8,
            "entity": pl.Utf8,
            "protein_id": pl.Utf8,
            "protein": pl.Utf8,
            "protein_organism": pl.Utf8,
            "peptide_start": pl.Utf8,
            "modified_sequence": pl.Utf8,
            "stripped_sequence": pl.Utf8,
            "charge": pl.Utf8,
            "charge_states": pl.Utf8,
            "precursor_id": pl.Utf8,
            "nterm_status": pl.Utf8,
            "peptide_length": pl.Float64,
            "score": pl.Float64,
            "intensity": pl.Float64,
            "summed_intensity": pl.Float64,
            "evidence_count": pl.Int64,
        }
    )


def empty_upset_evidence_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "item_id": pl.Utf8,
            "search": pl.Utf8,
            "protein_id": pl.Utf8,
            "protein_organism": pl.Utf8,
            "peptide_start": pl.Utf8,
            "modified_sequence": pl.Utf8,
            "stripped_sequence": pl.Utf8,
            "charge": pl.Utf8,
            "precursor_id": pl.Utf8,
            "nterm_status": pl.Utf8,
            "nterm_modif": pl.Utf8,
            "specificity": pl.Utf8,
            "Precursor.Normalised": pl.Float64,
        }
    )


def gravy(sequence: str) -> float | None:
    """Return Kyte-Doolittle GRAVY hydrophobicity for an amino-acid sequence."""

    scale = {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }
    values = [scale[aa] for aa in sequence if aa in scale]
    if not values:
        return None
    return sum(values) / len(values)
