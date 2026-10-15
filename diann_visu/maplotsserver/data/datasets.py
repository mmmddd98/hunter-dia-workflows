from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import polars as pl

from .set_analysis import (
    ENTITY_LEVEL_DESCRIPTIONS,
    ENTITY_LEVEL_LABELS,
    EVIDENCE_FILTER_LABELS,
    evidence_filter_label,
    UPSET_ITEM_LEVEL_DESCRIPTIONS,
    UPSET_ITEM_LEVEL_LABELS,
    EntityLevel,
    entity_feature_table,
    entity_sets,
    normalize_evidence_filter,
    normalize_evidence_filters,
    normalize_upset_item_level,
    prepare_upset_evidence_rows,
    prepare_upset_items,
    raw_with_termini_evidence,
)

from .repository import ExperimentRepository, sample_columns


ITEM_LEVEL_OPTIONS = [
    {"label": label, "value": value}
    for value, label in UPSET_ITEM_LEVEL_LABELS.items()
]
EVIDENCE_FILTER_OPTIONS = [
    {"label": label, "value": value}
    for value, label in EVIDENCE_FILTER_LABELS.items()
]
DEFAULT_ITEM_LEVEL = "nterm_site"
DEFAULT_EVIDENCE_FILTER = "relevant"
UPSET_AGGREGATION_SOURCE_OPTIONS = [
    {"label": "Materialized tables", "value": "materialized"},
    {"label": "On the fly", "value": "on_the_fly"},
]
DEFAULT_UPSET_AGGREGATION_SOURCE = "materialized"
TRIPLEMIX_ORGANISM_OPTIONS = [
    {"label": "All organisms", "value": "__all__"},
    {"label": "Homo sapiens", "value": "Homo sapiens"},
    {"label": "Arabidopsis thaliana", "value": "Arabidopsis thaliana"},
    {"label": "Escherichia coli", "value": "Escherichia coli"},
]

# Backwards-compatible names for old imports/callback ids while the UI wording
# changes from "source" to item level.
SOURCE_OPTIONS = ITEM_LEVEL_OPTIONS
SOURCE_LABELS = {option["value"]: option["label"] for option in SOURCE_OPTIONS}
MULTI_DATASET_PREFIX = "multi::"
MULTI_DATASET_SEPARATOR = ","


@dataclass
class DatasetRegistry:
    repository: ExperimentRepository

    def all(self) -> dict[str, dict]:
        experiments = self.repository.list_experiments()
        return {
            experiment: self._get_experiment(experiment)
            for experiment in experiments
        }

    def get(self, dataset_id: str) -> dict | None:
        if is_multi_dataset_id(dataset_id):
            return self.multi_get(dataset_id)
        if dataset_id not in set(self.repository.list_experiments()):
            return None
        return self._get_experiment(dataset_id)

    def _get_experiment(self, dataset_id: str) -> dict:
        searches = self.repository.list_searches(dataset_id)
        live_summary = self.repository.quick_live_summary(dataset_id)
        pipeline_version = self.repository.pipeline_version_for_experiment(dataset_id)
        return {
            "id": dataset_id,
            "title": display_title(dataset_id),
            "description": f"{len(searches)} DIA-NN search result(s) in pipeline {pipeline_version}.",
            "pipeline_version": pipeline_version,
            "cluster": self.repository.cluster_for_experiment(dataset_id),
            "organism": self.repository.organism_for_experiment(dataset_id),
            "experiment_type": self.repository.experiment_type_for_experiment(dataset_id),
            "has_entrapment": self.repository.has_entrapment_hits(dataset_id),
            "is_running": live_summary.get("is_live", False),
            "current_step": live_summary.get("current_step") if live_summary.get("is_live", False) else None,
            "set_columns": searches,
            "item_df": pd.DataFrame(),
        }

    def multi_get(self, dataset_id: str) -> dict | None:
        dataset_ids = multi_dataset_ids(dataset_id)
        datasets = [self.get(current_id) for current_id in dataset_ids]
        if len(dataset_ids) < 2 or any(dataset is None for dataset in datasets):
            return None
        valid_datasets = [dataset for dataset in datasets if dataset is not None]
        organisms = {dataset["organism"] for dataset in valid_datasets}
        if len(organisms) != 1:
            return None
        set_columns = [
            self.multi_search_label(current_id, search)
            for current_id in dataset_ids
            for search in self.repository.list_searches(current_id)
        ]
        return {
            "id": dataset_id,
            "title": f"Multi experiment UpSet ({len(dataset_ids)} experiments)",
            "description": f"{len(set_columns)} DIA-NN search result(s) across {len(dataset_ids)} experiments.",
            "pipeline_version": "multiple",
            "cluster": "multiple",
            "organism": valid_datasets[0]["organism"],
            "experiment_type": "multi experiment",
            "has_entrapment": any(dataset["has_entrapment"] for dataset in valid_datasets),
            "is_running": False,
            "current_step": None,
            "set_columns": set_columns,
            "item_df": pd.DataFrame(),
            "multi_dataset_ids": dataset_ids,
        }

    def organism_options(self) -> list[dict]:
        organisms = sorted({dataset["organism"] for dataset in self.all().values()})
        return [{"label": "All organisms", "value": "__all__"}] + [
            {"label": organism, "value": organism} for organism in organisms
        ]

    def experiment_type_options(self) -> list[dict]:
        experiment_types = sorted({dataset["experiment_type"] for dataset in self.all().values()})
        return [{"label": "All experiment types", "value": "__all__"}] + [
            {"label": experiment_type, "value": experiment_type} for experiment_type in experiment_types
        ]

    def build_upset_items(
        self,
        dataset_id: str,
        item_level: str | None = None,
        evidence_filter: str | list[str] | None = None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
        aggregation_source: str | None = None,
        searches: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[str], str]:
        if is_multi_dataset_id(dataset_id):
            return self.build_multi_upset_items(
                dataset_id,
                item_level,
                evidence_filter,
                include_entrapment=include_entrapment,
                qvalue_threshold=qvalue_threshold,
                aggregation_source=aggregation_source,
            )
        normalized_level = normalize_upset_item_level(item_level)
        normalized_filter = normalize_evidence_filters(evidence_filter)
        source = normalize_upset_aggregation_source(aggregation_source)
        if source == "materialized":
            item_df_pl, set_cols = self.repository.load_or_build_upset_items(
                dataset_id,
                normalized_level,
                normalized_filter,
                include_entrapment=include_entrapment,
                qvalue_threshold=qvalue_threshold,
                searches=searches,
            )
        else:
            item_df_pl, set_cols = self.repository.build_upset_items_on_the_fly(
                dataset_id,
                normalized_level,
                normalized_filter,
                include_entrapment=include_entrapment,
                qvalue_threshold=qvalue_threshold,
                searches=searches,
            )
        status_label = upset_status_label(normalized_level, normalized_filter)
        if item_df_pl.is_empty():
            return empty_item_df(), [], status_label
        item_df = item_df_pl.to_pandas()
        if include_entrapment and "is_entrapment" in item_df.columns:
            item_df["is_decoy"] = item_df["is_entrapment"].fillna(False).astype(bool)
        else:
            item_df = item_df.drop(columns=[col for col in ("is_entrapment", "id_origin", "is_decoy") if col in item_df.columns])
        for col in (
            "protein",
            "protein_id",
            "protein_organism",
            "peptide_start",
            "modified_sequence",
            "stripped_sequence",
            "charge",
            "charge_states",
            "precursor_id",
            "nterm_status",
            "peptide_length",
            "score",
            "intensity",
            "summed_intensity",
            "gravy",
        ):
            if col not in item_df.columns:
                item_df[col] = ""
        item_df["protein"] = item_df["protein"].fillna("")
        return item_df, set_cols, status_label

    def build_multi_upset_items(
        self,
        dataset_id: str,
        item_level: str | None = None,
        evidence_filter: str | list[str] | None = None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
        aggregation_source: str | None = None,
    ) -> tuple[pd.DataFrame, list[str], str]:
        dataset_ids = multi_dataset_ids(dataset_id)
        frames = []
        all_set_cols = []
        for current_id in dataset_ids:
            current_df, current_set_cols, _status_label = self.build_upset_items(
                current_id,
                item_level,
                evidence_filter,
                include_entrapment=include_entrapment and self.has_entrapment_hits(current_id),
                qvalue_threshold=qvalue_threshold,
                aggregation_source=aggregation_source,
            )
            rename_map = {search: self.multi_search_label(current_id, search) for search in current_set_cols}
            current_df = current_df.rename(columns=rename_map)
            renamed_set_cols = list(rename_map.values())
            all_set_cols.extend(renamed_set_cols)
            frames.append(current_df)

        if not frames:
            return empty_item_df(), [], upset_status_label(item_level or DEFAULT_ITEM_LEVEL, evidence_filter or DEFAULT_EVIDENCE_FILTER)

        for frame in frames:
            for set_col in all_set_cols:
                if set_col not in frame.columns:
                    frame[set_col] = False
            for set_col in all_set_cols:
                frame[set_col] = frame[set_col].fillna(False).astype(bool)

        combined = pd.concat(frames, ignore_index=True, sort=False)
        if combined.empty or "item_id" not in combined.columns:
            return empty_item_df(), all_set_cols, upset_status_label(item_level or DEFAULT_ITEM_LEVEL, evidence_filter or DEFAULT_EVIDENCE_FILTER)

        bool_metadata_cols = [
            col
            for col in ("is_entrapment", "is_decoy")
            if col in combined.columns
        ]
        agg = {set_col: "max" for set_col in all_set_cols}
        agg.update({col: "max" for col in bool_metadata_cols})
        for col in combined.columns:
            if col not in agg and col != "item_id":
                agg[col] = "first"
        combined = combined.groupby("item_id", as_index=False, dropna=False).agg(agg)
        for bool_col in [*all_set_cols, *bool_metadata_cols]:
            combined[bool_col] = combined[bool_col].fillna(False).astype(bool)
        status_label = upset_status_label(item_level or DEFAULT_ITEM_LEVEL, evidence_filter or DEFAULT_EVIDENCE_FILTER)
        return combined, all_set_cols, status_label

    def build_upset_evidence_rows(
        self,
        dataset_id: str,
        item_level: str | None = None,
        evidence_filter: str | list[str] | None = None,
        *,
        include_entrapment: bool = False,
        qvalue_threshold: float | None = None,
    ) -> pd.DataFrame:
        if is_multi_dataset_id(dataset_id):
            frames = []
            for current_id in multi_dataset_ids(dataset_id):
                current = self.build_upset_evidence_rows(
                    current_id,
                    item_level,
                    evidence_filter,
                    include_entrapment=include_entrapment and self.has_entrapment_hits(current_id),
                    qvalue_threshold=qvalue_threshold,
                )
                if current.empty:
                    continue
                current = current.copy()
                current["dataset_id"] = current_id
                current["search"] = current["search"].map(lambda search: self.multi_search_label(current_id, search))
                frames.append(current)
            return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=["item_id", "search", "stripped_sequence"])
        evidence = self.repository.load_or_build_upset_evidence_rows(
            dataset_id,
            normalize_upset_item_level(item_level),
            normalize_evidence_filters(evidence_filter),
            include_entrapment=include_entrapment,
            qvalue_threshold=qvalue_threshold,
        )
        return evidence.to_pandas() if not evidence.is_empty() else pd.DataFrame(columns=["item_id", "search", "stripped_sequence"])

    def default_source(self, dataset_id: str) -> str:
        return DEFAULT_ITEM_LEVEL

    def default_item_level(self, dataset_id: str) -> str:
        return DEFAULT_ITEM_LEVEL

    def default_upset_aggregation_source(self, dataset_id: str) -> str:
        return DEFAULT_UPSET_AGGREGATION_SOURCE

    def raw_tables(self, dataset_id: str, *, include_entrapment: bool = False) -> dict[str, pl.DataFrame]:
        if is_multi_dataset_id(dataset_id):
            return self.multi_tables(dataset_id, lambda current_id: self.raw_tables(current_id, include_entrapment=include_entrapment))
        return self.repository.load_all_raw(dataset_id, include_entrapment=include_entrapment)

    def has_entrapment_hits(self, dataset_id: str) -> bool:
        if is_multi_dataset_id(dataset_id):
            return any(self.has_entrapment_hits(current_id) for current_id in multi_dataset_ids(dataset_id))
        return self.repository.has_entrapment_hits(dataset_id)

    def max_reported_qvalue(self, dataset_id: str) -> float | None:
        if is_multi_dataset_id(dataset_id):
            values = [
                self.max_reported_qvalue(current_id)
                for current_id in multi_dataset_ids(dataset_id)
            ]
            values = [value for value in values if value is not None]
            return max(values) if values else None
        return self.repository.max_reported_qvalue(dataset_id)

    def fasta_path_for_dataset(self, dataset_id: str):
        if is_multi_dataset_id(dataset_id):
            for current_id in multi_dataset_ids(dataset_id):
                path = self.repository.fasta_path_for_experiment(current_id)
                if path is not None:
                    return path
            return None
        return self.repository.fasta_path_for_experiment(dataset_id)

    def multi_search_label(self, dataset_id: str, search: str) -> str:
        return f"{display_title(dataset_id)} / {search}"

    def adapted_tables(self, dataset_id: str) -> dict[str, pl.DataFrame]:
        if is_multi_dataset_id(dataset_id):
            return self.multi_tables(dataset_id, self.adapted_tables)
        return self.repository.load_all_adapted(dataset_id)

    def termini_tables(self, dataset_id: str) -> dict[str, pl.DataFrame]:
        if is_multi_dataset_id(dataset_id):
            return self.multi_tables(dataset_id, self.termini_tables)
        return self.repository.load_all_termini(dataset_id)

    def multi_tables(self, dataset_id: str, loader) -> dict[str, pl.DataFrame]:
        tables: dict[str, pl.DataFrame] = {}
        for current_id in multi_dataset_ids(dataset_id):
            for search, table in loader(current_id).items():
                tables[self.multi_search_label(current_id, search)] = table
        return tables


def source_tables_for_upset(
    *,
    raw_tables: dict[str, pl.DataFrame],
    adapted_tables: dict[str, pl.DataFrame],
    termini_tables: dict[str, pl.DataFrame],
    source: str | None,
) -> tuple[dict[str, pl.DataFrame], EntityLevel, str]:
    entity_level = normalize_source_level(source)
    if entity_level == "nterm":
        tables = termini_tables or adapted_tables or raw_tables
    elif entity_level == "peptidoform":
        tables = raw_tables
    elif entity_level == "protein":
        tables = raw_tables or termini_tables or adapted_tables
    else:
        tables = raw_tables or termini_tables or adapted_tables
    return tables, entity_level, ENTITY_LEVEL_LABELS[entity_level]


def filter_upset_items_by_organism(item_df: pd.DataFrame, organism: str | None) -> pd.DataFrame:
    """Filter protein-keyed UpSet items by FASTA organism annotation."""
    if item_df.empty or organism in (None, "", "__all__"):
        return item_df
    if "protein_organism" not in item_df.columns:
        return item_df.iloc[0:0].copy()
    values = item_df["protein_organism"].fillna("").astype(str)
    return item_df.loc[values == str(organism)].copy()


def normalize_source_level(source: str | None) -> EntityLevel:
    aliases: dict[str, EntityLevel] = {
        "termini_nterm": "nterm",
        "adapted_nterm": "nterm",
        "termini_peptide": "peptide",
        "adapted_peptide": "peptide",
        "raw_peptide": "peptide",
        "raw_peptidoform": "peptidoform",
        "raw_protein": "protein",
    }
    if source in {"nterm", "peptide", "peptidoform", "protein"}:
        return source  # type: ignore[return-value]
    return aliases.get(source or "", "nterm")


def normalize_upset_aggregation_source(value: str | None) -> str:
    if value in {"materialized", "on_the_fly"}:
        return value
    return DEFAULT_UPSET_AGGREGATION_SOURCE


def evidence_tables_for_upset(
    *,
    searches: list[str],
    raw_tables: dict[str, pl.DataFrame],
    adapted_tables: dict[str, pl.DataFrame],
    termini_tables: dict[str, pl.DataFrame],
    include_entrapment: bool = False,
) -> dict[str, pl.DataFrame]:
    tables = {}
    for search in searches:
        if search in raw_tables and search in termini_tables:
            tables[search] = raw_with_termini_evidence(raw_tables[search], termini_tables[search], include_entrapment=include_entrapment)
        elif search in termini_tables:
            tables[search] = termini_tables[search]
        elif search in adapted_tables:
            tables[search] = adapted_tables[search]
        elif search in raw_tables:
            tables[search] = raw_tables[search]
    return tables


def source_status_label(source_label: str, entity_level: EntityLevel) -> str:
    level_label = ENTITY_LEVEL_LABELS[entity_level]
    meaning = ENTITY_LEVEL_DESCRIPTIONS[entity_level]
    return f"{level_label}: {meaning}"


def upset_status_label(item_level: str, evidence_filter: str | list[str] | tuple[str, ...]) -> str:
    level = normalize_upset_item_level(item_level)
    return f"{UPSET_ITEM_LEVEL_LABELS[level]}: {UPSET_ITEM_LEVEL_DESCRIPTIONS[level]} | {evidence_filter_label(evidence_filter)}"


def feature_metadata(feature_df: pl.DataFrame) -> pd.DataFrame:
    if feature_df.is_empty() or "entity" not in feature_df.columns:
        return pd.DataFrame()

    exprs = []
    if "protein" in feature_df.columns:
        exprs.append(text_expr("protein").drop_nulls().first().alias("protein"))
    else:
        exprs.append(pl.lit("").alias("protein"))
    if "length" in feature_df.columns:
        exprs.append(pl.col("length").cast(pl.Int64, strict=False).median().alias("peptide_length"))
    if "Q.Value" in feature_df.columns:
        exprs.append(pl.col("Q.Value").cast(pl.Float64, strict=False).min().alias("score"))
    elif "PEP" in feature_df.columns:
        exprs.append(pl.col("PEP").cast(pl.Float64, strict=False).min().alias("score"))
    if "abundance" in feature_df.columns:
        exprs.append(pl.col("abundance").cast(pl.Float64, strict=False).median().alias("intensity"))
    for candidate in (
        "RT",
        "Predicted.RT",
        "Abs.RT.Delta",
        "Best.Fr.Mz.Delta",
        "Ms1.Apex.Mz.Delta",
        "Abs.Best.Fr.Mz.Delta",
        "Abs.Ms1.Apex.Mz.Delta",
        "gravy",
    ):
        if candidate in feature_df.columns:
            exprs.append(pl.col(candidate).cast(pl.Float64, strict=False).median().alias(safe_column_name(candidate)))
    for candidate in (
        "sequence",
        "nterm_modif",
        "specificity",
        "nterm_residue",
        "cterm_residue",
        "nterm_motif3",
        "cterm_motif3",
        "id_origin",
    ):
        if candidate in feature_df.columns:
            exprs.append(text_expr(candidate).drop_nulls().first().alias(candidate))
    if "is_entrapment" in feature_df.columns:
        exprs.append(pl.col("is_entrapment").cast(pl.Boolean, strict=False).max().alias("is_entrapment"))

    metadata = feature_df.group_by("entity").agg(exprs).to_pandas()
    for col in (
        "peptide_length",
        "score",
        "intensity",
        "RT",
        "Predicted_RT",
        "Abs_RT_Delta",
        "Best_Fr_Mz_Delta",
        "Ms1_Apex_Mz_Delta",
        "Abs_Best_Fr_Mz_Delta",
        "Abs_Ms1_Apex_Mz_Delta",
        "gravy",
        "sequence",
        "nterm_modif",
        "specificity",
        "nterm_residue",
        "cterm_residue",
        "nterm_motif3",
        "cterm_motif3",
        "id_origin",
        "is_entrapment",
    ):
        if col not in metadata.columns:
            metadata[col] = None
    return metadata[
        [
            "entity",
            "protein",
            "peptide_length",
            "score",
            "intensity",
            "RT",
            "Predicted_RT",
            "Abs_RT_Delta",
            "Best_Fr_Mz_Delta",
            "Ms1_Apex_Mz_Delta",
            "Abs_Best_Fr_Mz_Delta",
            "Abs_Ms1_Apex_Mz_Delta",
            "gravy",
            "sequence",
            "nterm_modif",
            "specificity",
            "nterm_residue",
            "cterm_residue",
            "nterm_motif3",
            "cterm_motif3",
            "id_origin",
            "is_entrapment",
        ]
    ]


def empty_item_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "item_id",
            "entity",
            "protein",
            "protein_id",
            "peptide_start",
            "modified_sequence",
            "stripped_sequence",
            "charge",
            "charge_states",
            "precursor_id",
            "nterm_status",
            "peptide_length",
            "score",
            "intensity",
            "summed_intensity",
        ]
    )


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


def display_title(dataset_id: str) -> str:
    parts = dataset_id.split("::")
    if len(parts) >= 3:
        return f"{parts[-1].replace('_', ' ')} ({parts[-2]})"
    if len(parts) == 2:
        return parts[-1].replace("_", " ")
    return dataset_id.replace("_", " ")


def create_multi_dataset_id(dataset_ids: list[str] | tuple[str, ...]) -> str:
    return f"{MULTI_DATASET_PREFIX}{MULTI_DATASET_SEPARATOR.join(dataset_ids)}"


def is_multi_dataset_id(dataset_id: str | None) -> bool:
    return bool(dataset_id and dataset_id.startswith(MULTI_DATASET_PREFIX))


def multi_dataset_ids(dataset_id: str) -> list[str]:
    if not is_multi_dataset_id(dataset_id):
        return [dataset_id]
    payload = dataset_id.removeprefix(MULTI_DATASET_PREFIX)
    return [part for part in payload.split(MULTI_DATASET_SEPARATOR) if part]


def summarize_quant_tables(adapted_tables: dict[str, pl.DataFrame]) -> pd.DataFrame:
    rows = []
    for search, df in adapted_tables.items():
        denom = max(df.height, 1)
        for col in sample_columns(df):
            present = df.select(pl.col(col).is_not_null().sum()).item()
            rows.append({"search": search, "sample": col, "fraction_present": present / denom})
    return pd.DataFrame(rows)
