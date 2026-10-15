from __future__ import annotations

from collections.abc import Iterable

import polars as pl


ENTRAPMENT_PREFIX = "ENT_"
PROTEIN_ID_COLUMNS = ("Protein.Ids", "Protein.Group", "Protein.Names", "Genes", "protein", "protein_id")
DECOY_COLUMNS = ("Decoy", "decoy", "Is.Decoy", "is_decoy", "is_diann_decoy")
NTERM_LEVEL_TO_ID_PARTS = {
    "nterm_precursor": ("protein_id", "peptide_start", "stripped_sequence", "nterm_status", "charge"),
    "nterm_peptide": ("protein_id", "peptide_start", "stripped_sequence", "nterm_status"),
    "nterm_site": ("protein_id", "peptide_start", "nterm_status"),
    "nterm_protein": ("protein_id",),
}
QVALUE_COLUMNS_BY_LEVEL = {
    "precursor": ("Q.Value", "Global.Q.Value", "Lib.Q.Value"),
    "peptidoform": ("Peptidoform.Q.Value", "Global.Peptidoform.Q.Value", "Lib.Peptidoform.Q.Value", "Q.Value"),
    "peptide": ("Q.Value", "Global.Q.Value", "Lib.Q.Value"),
    "protein_group": ("PG.Q.Value", "Global.PG.Q.Value", "Lib.PG.Q.Value", "Protein.Q.Value"),
    "protein": ("Protein.Q.Value", "PG.Q.Value", "Global.PG.Q.Value", "Lib.PG.Q.Value"),
    "nterm_precursor": ("Q.Value",),
    "nterm_peptide": ("Q.Value",),
    "nterm_site": ("Q.Value",),
    "nterm_protein": ("Q.Value",),
}


def entrapment_hit_expr(columns: Iterable[str] | None = None, prefix: str = ENTRAPMENT_PREFIX) -> pl.Expr:
    """Return a Polars expression identifying rows assigned to entrapment proteins."""
    cols = tuple(PROTEIN_ID_COLUMNS if columns is None else columns)
    expr: pl.Expr | None = None
    pattern = rf"(^|[;,\s]){prefix}"
    for col in cols:
        part = (
            pl.col(col)
            .cast(pl.Utf8, strict=False)
            .str.contains(pattern)
            .fill_null(False)
        )
        expr = part if expr is None else (expr | part)
    return expr if expr is not None else pl.lit(False)


def decoy_hit_expr(columns: Iterable[str] | None = None) -> pl.Expr:
    """Return a Polars expression identifying DIA-NN internal decoy rows."""
    cols = tuple(DECOY_COLUMNS if columns is None else columns)
    expr: pl.Expr | None = None
    for col in cols:
        part = pl.col(col).cast(pl.Int64, strict=False).fill_null(0) != 0
        expr = part if expr is None else (expr | part)
    return expr if expr is not None else pl.lit(False)


def primary_entrapment_hit_expr(columns: Iterable[str], prefix: str = ENTRAPMENT_PREFIX) -> pl.Expr:
    cols = set(columns)
    for col in ("Protein.Group", "Protein.Ids", "protein"):
        if col in cols:
            return pl.col(col).cast(pl.Utf8, strict=False).str.starts_with(prefix).fill_null(False)
    return entrapment_hit_expr(columns, prefix=prefix)


def add_entrapment_columns(df: pl.DataFrame) -> pl.DataFrame:
    protein_columns = [col for col in PROTEIN_ID_COLUMNS if col in df.columns]
    decoy_columns = [col for col in DECOY_COLUMNS if col in df.columns]
    return df.with_columns(
        primary_entrapment_hit_expr(protein_columns).alias("is_entrapment"),
        decoy_hit_expr(decoy_columns).alias("is_diann_decoy"),
    ).with_columns(
        pl.when(pl.col("is_diann_decoy"))
        .then(pl.lit("diann_decoy"))
        .when(pl.col("is_entrapment"))
        .then(pl.lit("entrapment"))
        .otherwise(pl.lit("target"))
        .alias("id_origin")
    )


def exclude_entrapment(df: pl.DataFrame) -> pl.DataFrame:
    annotated = add_entrapment_columns(df)
    return annotated.filter(~pl.col("is_entrapment"), ~pl.col("is_diann_decoy")).drop(
        "is_entrapment", "is_diann_decoy", "id_origin"
    )


def exclude_diann_decoys(df: pl.DataFrame) -> pl.DataFrame:
    annotated = add_entrapment_columns(df)
    return annotated.filter(~pl.col("is_diann_decoy"))


def entity_column_for_level(df: pl.DataFrame, level: str) -> str | None:
    candidates = {
        "precursor": ("Precursor.Id", "precursor_id", "Modified.Sequence", "modified_sequence", "nterm_modif_peptide"),
        "peptidoform": ("Modified.Sequence", "modified_sequence", "Precursor.Id", "precursor_id", "nterm_modif_peptide"),
        "peptide": ("Stripped.Sequence", "stripped_sequence", "peptide"),
        "protein_group": ("Protein.Group", "Protein.Ids", "protein", "protein_id"),
    }.get(level, (level,))
    return next((col for col in candidates if col in df.columns), None)


def qvalue_column_for_level(df: pl.DataFrame, level: str, requested: str | None = None) -> str | None:
    if requested is not None:
        return requested if requested in df.columns else None
    candidates = QVALUE_COLUMNS_BY_LEVEL.get(level, ("Q.Value",))
    return next((col for col in candidates if col in df.columns), None)


def entity_selection_for_level(df: pl.DataFrame, level: str, qvalue_col: str | None = None) -> pl.DataFrame | None:
    resolved_qvalue_col = qvalue_column_for_level(df, level, qvalue_col)
    if resolved_qvalue_col is None:
        return None
    parts = NTERM_LEVEL_TO_ID_PARTS.get(level)
    if parts is not None:
        if not all(part in df.columns for part in parts):
            return None
        required = [
            pl.col(part).is_not_null() & (pl.col(part).cast(pl.Utf8, strict=False).str.strip_chars() != "")
            for part in parts
        ]
        valid = required[0]
        for expr in required[1:]:
            valid = valid & expr
        return (
            df.filter(valid)
            .select(
                pl.concat_str([pl.col(part).cast(pl.Utf8) for part in parts], separator="|").alias("entity"),
                pl.col(resolved_qvalue_col).cast(pl.Float64, strict=False).alias("qvalue"),
                pl.col("is_entrapment"),
                pl.col("is_diann_decoy"),
            )
            .drop_nulls(["entity", "qvalue"])
        )
    entity_col = entity_column_for_level(df, level)
    if entity_col is None:
        return None
    return (
        df.select(
            pl.col(entity_col).cast(pl.Utf8).alias("entity"),
            pl.col(resolved_qvalue_col).cast(pl.Float64, strict=False).alias("qvalue"),
            pl.col("is_entrapment"),
            pl.col("is_diann_decoy"),
        )
        .drop_nulls(["entity", "qvalue"])
    )


def entrapment_summary(
    tables: dict[str, pl.DataFrame],
    *,
    levels: tuple[str, ...] = ("precursor", "peptidoform", "peptide", "protein_group"),
    qvalue_threshold: float | None = None,
    compare_diann_decoys: bool = False,
) -> pl.DataFrame:
    rows = []
    for search, df in tables.items():
        annotated = add_entrapment_columns(df) if compare_diann_decoys else exclude_diann_decoys(df)
        for level in levels:
            if qvalue_threshold is None:
                entity_col = entity_column_for_level(annotated, level)
                if entity_col is None:
                    continue
                counts = (
                    annotated.select(
                        pl.col(entity_col).cast(pl.Utf8).alias("entity"),
                        pl.col("id_origin"),
                    )
                    .drop_nulls("entity")
                    .unique()
                    .group_by("id_origin")
                    .len()
                )
            else:
                selected = entity_selection_for_level(annotated, level)
                if selected is None:
                    continue
                best = (
                    selected
                    .group_by("entity")
                    .agg(
                        pl.col("qvalue").min().alias("qvalue"),
                        pl.col("is_entrapment").max().alias("is_entrapment"),
                        pl.col("is_diann_decoy").max().alias("is_diann_decoy"),
                    )
                    .filter((pl.col("qvalue") >= 0) & (pl.col("qvalue") <= float(qvalue_threshold)))
                    .with_columns(
                        pl.when(compare_diann_decoys and pl.col("is_diann_decoy"))
                        .then(pl.lit("diann_decoy"))
                        .when(pl.col("is_entrapment"))
                        .then(pl.lit("entrapment"))
                        .otherwise(pl.lit("target"))
                        .alias("id_origin"),
                    )
                )
                counts = best.group_by("id_origin").len() if not best.is_empty() else pl.DataFrame({"id_origin": [], "len": []})
            count_by_origin = {row["id_origin"]: int(row["len"]) for row in counts.to_dicts()}
            target = count_by_origin.get("target", 0)
            entrapment = count_by_origin.get("entrapment", 0)
            diann_decoy = count_by_origin.get("diann_decoy", 0)
            total = target + entrapment + (diann_decoy if compare_diann_decoys else 0)
            row = {
                "search": search,
                "level": level,
                "target_ids": target,
                "entrapment_ids": entrapment,
                "total_ids": total,
                "entrapment_fraction": entrapment / (target + entrapment) if target + entrapment else 0.0,
                "entrapment_fdr_estimate": entrapment / target if target else 0.0,
            }
            if compare_diann_decoys:
                row["diann_decoy_ids"] = diann_decoy
                row["diann_decoy_fdr_estimate"] = diann_decoy / target if target else 0.0
            rows.append(row)
    return pl.DataFrame(rows)


def entrapment_threshold_metrics(
    tables: dict[str, pl.DataFrame],
    *,
    level: str = "peptide",
    qvalue_threshold: float = 0.01,
    qvalue_col: str | None = None,
    compare_diann_decoys: bool = False,
) -> pl.DataFrame:
    rows = []
    for search, df in tables.items():
        annotated = add_entrapment_columns(df) if compare_diann_decoys else exclude_diann_decoys(df)
        selected = entity_selection_for_level(annotated, level, qvalue_col)
        if selected is None:
            continue
        agg_exprs = [
            pl.col("qvalue").min().alias("qvalue"),
            pl.col("is_entrapment").max().alias("is_entrapment"),
        ]
        if compare_diann_decoys:
            agg_exprs.append(pl.col("is_diann_decoy").max().alias("is_diann_decoy"))
        best = (
            selected
            .group_by("entity")
            .agg(agg_exprs)
            .filter((pl.col("qvalue") >= 0) & (pl.col("qvalue") <= qvalue_threshold))
            .with_columns(
                pl.when(compare_diann_decoys and pl.col("is_diann_decoy"))
                .then(pl.lit("diann_decoy"))
                .when(pl.col("is_entrapment"))
                .then(pl.lit("entrapment"))
                .otherwise(pl.lit("target"))
                .alias("id_origin"),
            )
        )
        if best.is_empty():
            target = entrapment = diann_decoy = 0
        else:
            counts = best.group_by("id_origin").len()
            count_by_origin = {row["id_origin"]: int(row["len"]) for row in counts.to_dicts()}
            target = count_by_origin.get("target", 0)
            entrapment = count_by_origin.get("entrapment", 0)
            diann_decoy = count_by_origin.get("diann_decoy", 0)
        rows.append(
            {
                "search": search,
                "level": level,
                "qvalue_threshold": float(qvalue_threshold),
                "n_target": target,
                "n_entrapment": entrapment,
                "n_program_decoy": diann_decoy,
                "total_ids": target + entrapment + diann_decoy,
                "entrapment_rate": entrapment / target if target else 0.0,
                "entrapment_fraction": entrapment / (target + entrapment) if target + entrapment else 0.0,
                "program_decoy_rate": diann_decoy / target if target else 0.0,
            }
        )
    return pl.DataFrame(rows)


def entrapment_qvalue_curve(
    tables: dict[str, pl.DataFrame],
    *,
    level: str = "peptide",
    qvalue_col: str | None = None,
    compare_diann_decoys: bool = False,
) -> pl.DataFrame:
    frames = []
    for search, df in tables.items():
        annotated = add_entrapment_columns(df) if compare_diann_decoys else exclude_diann_decoys(df)
        selected = entity_selection_for_level(annotated, level, qvalue_col)
        if selected is None:
            continue
        agg_exprs = [
            pl.col("qvalue").min().alias("qvalue"),
            pl.col("is_entrapment").max().alias("is_entrapment"),
        ]
        if compare_diann_decoys:
            agg_exprs.append(pl.col("is_diann_decoy").max().alias("is_diann_decoy"))
        best = (
            selected
            .sort("qvalue")
            .group_by("entity")
            .agg(agg_exprs)
            .sort("qvalue")
            .with_row_index("rank", offset=1)
        )
        if compare_diann_decoys:
            best = best.with_columns(
                (pl.col("is_entrapment") & ~pl.col("is_diann_decoy")).cum_sum().alias("cum_entrapment"),
                pl.col("is_diann_decoy").cum_sum().alias("cum_diann_decoy"),
                (~pl.col("is_entrapment") & ~pl.col("is_diann_decoy")).cast(pl.UInt32).cum_sum().alias("cum_target"),
                pl.lit(search).alias("search"),
                pl.lit(level).alias("level"),
            ).with_columns(
                (pl.col("cum_entrapment") / (pl.col("cum_target") + pl.col("cum_entrapment"))).alias("entrapment_fraction"),
                pl.when(pl.col("cum_target") > 0).then(pl.col("cum_entrapment") / pl.col("cum_target")).otherwise(0.0).alias("entrapment_fdr_estimate"),
                pl.when(pl.col("cum_target") > 0).then(pl.col("cum_diann_decoy") / pl.col("cum_target")).otherwise(0.0).alias("diann_decoy_fdr_estimate"),
                pl.col("qvalue").alias("diann_reported_fdr"),
            ).select(
                "search",
                "level",
                "rank",
                "qvalue",
                "cum_target",
                "cum_entrapment",
                "cum_diann_decoy",
                "entrapment_fraction",
                "entrapment_fdr_estimate",
                "diann_decoy_fdr_estimate",
                "diann_reported_fdr",
            )
            best = best.unpivot(
                index=["search", "level", "rank", "qvalue", "cum_target", "cum_entrapment", "cum_diann_decoy", "entrapment_fraction"],
                on=["entrapment_fdr_estimate", "diann_reported_fdr"],
                variable_name="method",
                value_name="fdr_estimate",
            ).with_columns(
                pl.col("method").replace(
                    {
                        "entrapment_fdr_estimate": "ENT_ entrapment",
                        "diann_reported_fdr": "DIA-NN Q.Value threshold",
                    }
                )
            )
        else:
            best = best.with_columns(
                pl.col("is_entrapment").cum_sum().alias("cum_entrapment"),
                (~pl.col("is_entrapment")).cast(pl.UInt32).cum_sum().alias("cum_target"),
                pl.lit(search).alias("search"),
                pl.lit(level).alias("level"),
            ).with_columns(
                (pl.col("cum_entrapment") / (pl.col("cum_target") + pl.col("cum_entrapment"))).alias("entrapment_fraction"),
                pl.when(pl.col("cum_target") > 0).then(pl.col("cum_entrapment") / pl.col("cum_target")).otherwise(0.0).alias("entrapment_fdr_estimate"),
            ).with_columns(pl.col("entrapment_fdr_estimate").alias("fdr_estimate")).select(
                "search",
                "level",
                "rank",
                "qvalue",
                "cum_target",
                "cum_entrapment",
                "entrapment_fraction",
                "entrapment_fdr_estimate",
                "fdr_estimate",
            )
        frames.append(best)
    if not frames:
        return pl.DataFrame(
            schema={
                "search": pl.Utf8,
                "level": pl.Utf8,
                "rank": pl.UInt32,
                "qvalue": pl.Float64,
                "cum_target": pl.UInt32,
                "cum_entrapment": pl.UInt32,
                "entrapment_fraction": pl.Float64,
                "entrapment_fdr_estimate": pl.Float64,
                "fdr_estimate": pl.Float64,
            }
        )
    return pl.concat(frames, how="vertical")


def entrapment_qvalue_distribution(
    tables: dict[str, pl.DataFrame],
    *,
    level: str = "peptide",
    qvalue_col: str | None = None,
    max_qvalue: float | None = None,
    bin_width: float = 0.01,
    compare_diann_decoys: bool = False,
) -> pl.DataFrame:
    frames = []
    for search, df in tables.items():
        annotated = add_entrapment_columns(df) if compare_diann_decoys else exclude_diann_decoys(df)
        selected = entity_selection_for_level(annotated, level, qvalue_col)
        if selected is None:
            continue
        agg_exprs = [
            pl.col("qvalue").min().alias("qvalue"),
            pl.col("is_entrapment").max().alias("is_entrapment"),
        ]
        if compare_diann_decoys:
            agg_exprs.append(pl.col("is_diann_decoy").max().alias("is_diann_decoy"))
        max_q_expr = pl.lit(float(max_qvalue)) if max_qvalue is not None else pl.col("qvalue").max()
        best = (
            selected
            .group_by("entity")
            .agg(agg_exprs)
            .filter((pl.col("qvalue") >= 0) & (pl.col("qvalue") <= max_q_expr))
            .with_columns(
                ((pl.col("qvalue") / bin_width).floor() * bin_width).alias("qvalue_bin_start"),
            )
            .with_columns(
                (pl.col("qvalue_bin_start") + bin_width).alias("qvalue_bin_end"),
                pl.when(compare_diann_decoys and pl.col("is_diann_decoy"))
                .then(pl.lit("diann_decoy"))
                .when(pl.col("is_entrapment"))
                .then(pl.lit("entrapment"))
                .otherwise(pl.lit("target"))
                .alias("id_origin"),
            )
        )
        if best.is_empty():
            continue
        binned = (
            best.group_by("qvalue_bin_start", "qvalue_bin_end", "id_origin")
            .len()
            .pivot(index=["qvalue_bin_start", "qvalue_bin_end"], columns="id_origin", values="len")
            .fill_null(0)
        )
        origins = ("target", "entrapment", "diann_decoy") if compare_diann_decoys else ("target", "entrapment")
        for col in origins:
            if col not in binned.columns:
                binned = binned.with_columns(pl.lit(0).alias(col))
        if "diann_decoy" not in binned.columns:
            binned = binned.with_columns(pl.lit(0).alias("diann_decoy"))
        binned = binned.with_columns(
            pl.lit(search).alias("search"),
            pl.lit(level).alias("level"),
            (pl.col("target") + pl.col("entrapment") + pl.col("diann_decoy")).alias("total_ids"),
            pl.when(pl.col("target") > 0)
            .then(pl.col("entrapment") / pl.col("target"))
            .otherwise(0.0)
            .alias("bin_entrapment_fdr_estimate"),
            pl.when(pl.col("target") > 0)
            .then(pl.col("diann_decoy") / pl.col("target"))
            .otherwise(0.0)
            .alias("bin_diann_decoy_fdr_estimate"),
            pl.col("qvalue_bin_end").alias("bin_diann_reported_qvalue"),
            pl.when((pl.col("target") + pl.col("entrapment")) > 0)
            .then(pl.col("entrapment") / (pl.col("target") + pl.col("entrapment")))
            .otherwise(0.0)
            .alias("bin_entrapment_fraction"),
        ).sort("qvalue_bin_start")
        frames.append(
            binned.select(
                "search",
                "level",
                "qvalue_bin_start",
                "qvalue_bin_end",
                "target",
                "entrapment",
                "diann_decoy",
                "total_ids",
                "bin_entrapment_fraction",
                "bin_entrapment_fdr_estimate",
                "bin_diann_decoy_fdr_estimate",
                "bin_diann_reported_qvalue",
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "search": pl.Utf8,
                "level": pl.Utf8,
                "qvalue_bin_start": pl.Float64,
                "qvalue_bin_end": pl.Float64,
                "target": pl.UInt32,
                "entrapment": pl.UInt32,
                "diann_decoy": pl.UInt32,
                "total_ids": pl.UInt32,
                "bin_entrapment_fraction": pl.Float64,
                "bin_entrapment_fdr_estimate": pl.Float64,
                "bin_diann_decoy_fdr_estimate": pl.Float64,
                "bin_diann_reported_qvalue": pl.Float64,
            }
        )
    return pl.concat(frames, how="vertical")
