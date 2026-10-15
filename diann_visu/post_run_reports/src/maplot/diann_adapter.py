from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Literal

import polars as pl

from .entrapment import exclude_entrapment


DEFAULT_UNIMOD_MAPPING: dict[str, str] = {
    "UniMod:1": "Acetyl",
    "UniMod:2016": "TMT",
    "UniMod:737": "TMT",
    "UniMod:738": "TMT",
    "UniMod:259": "TMT",
    "UniMod:26": "Pyro-Glu",
    "UniMod:27": "Pyro-Glu",
    "UniMod:34": "Methyl",
    "UniMod:35": "Oxidation",
    "UniMod:36": "Dimethyl",
    "UniMod:4": "Carbamidomethyl",
    "UniMod:510": "Dimethyl",
}


@dataclass
class DiannAdapter:
    """
    Convert DIA-NN result rows into a normalized peptide-by-sample matrix.

    Attributes:
        proteotypic: Keep only proteotypic entries when column is available.
        summarization: Aggregate duplicates per peptide/sample via "SUM" or "MAX".
        qvalue_threshold: Max precursor Q.Value allowed.
        protein_q_threshold: Max Protein.Q.Value allowed.
        pg_q_threshold: Max PG.Q.Value allowed.
        gg_q_threshold: Max GG.Q.Value allowed.
        significant_digits: Significant digits for optional post-rounding.
        unimod_mapping: UniMod token-to-name mapping used for N-term labels.
        quantity_column: DIA-NN quantity field to aggregate.
        apply_mad_scaling: Apply the legacy per-sample MAD transformation after log2.
    """
    proteotypic: bool = True
    summarization: Literal["SUM", "MAX"] = "SUM"
    qvalue_threshold: float = 0.01
    protein_q_threshold: float = 1.0
    pg_q_threshold: float = 1.0
    gg_q_threshold: float = 1.0
    significant_digits: int = 15
    unimod_mapping: Mapping[str, str] = field(default_factory=lambda: DEFAULT_UNIMOD_MAPPING)
    exclude_entrapment: bool = True
    quantity_column: str = "Precursor.Normalised"
    apply_mad_scaling: bool = True

    @staticmethod
    def expand_protein_ids(
        df: pl.DataFrame,
        key_columns: tuple[str, ...] = ("Stripped.Sequence",),
        protein_ids_column: str = "Protein.Ids",
    ) -> pl.DataFrame:
        """Expand DIA-NN's semicolon-delimited protein mappings without filtering.

        This is the mapping-only adapter operation. It deliberately does not
        apply the adapter's proteotypic, Q-value, signal, or normalization rules.
        """
        missing = set(key_columns + (protein_ids_column,)).difference(df.columns)
        if missing:
            raise ValueError(f"Missing columns for protein mapping expansion: {sorted(missing)}")
        return (
            df.select(*key_columns, protein_ids_column)
            .with_columns(pl.col(protein_ids_column).cast(pl.Utf8).str.split(";").alias("protein"))
            .explode("protein")
            .with_columns(
                pl.col("protein")
                .str.strip_chars()
                .str.replace(r" .*$", "")
                .alias("protein")
            )
            .filter(pl.col("protein").is_not_null() & (pl.col("protein") != ""))
            .drop(protein_ids_column)
            .unique(maintain_order=True)
        )

    def adapt(self, df: pl.DataFrame) -> pl.DataFrame:
        work = self._normalize_input_columns(df)

        required = {
            "File.Name",
            "Stripped.Sequence",
            "Modified.Sequence",
            "Protein.Ids",
            self.quantity_column,
            "Q.Value",
            "Protein.Q.Value",
            "PG.Q.Value",
            "GG.Q.Value",
        }
        missing = sorted(required.difference(work.columns))
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"DIA-NN adapter missing required columns: {missing_str}")

        work = work.with_columns(
            pl.col("Modified.Sequence").cast(pl.Utf8).alias("Modified.Sequence"),
            pl.col("Stripped.Sequence").cast(pl.Utf8).alias("Stripped.Sequence"),
            pl.col("Protein.Ids").cast(pl.Utf8).alias("Protein.Ids"),
            pl.col("File.Name").cast(pl.Utf8).alias("File.Name"),
            pl.col(self.quantity_column).cast(pl.Float64).alias(self.quantity_column),
            pl.col("Q.Value").cast(pl.Float64).alias("Q.Value"),
            pl.col("Protein.Q.Value").cast(pl.Float64).alias("Protein.Q.Value"),
            pl.col("PG.Q.Value").cast(pl.Float64).alias("PG.Q.Value"),
            pl.col("GG.Q.Value").cast(pl.Float64).alias("GG.Q.Value"),
        )

        if self.proteotypic and "Proteotypic" in work.columns:
            work = work.filter(pl.col("Proteotypic") != 0)

        if self.exclude_entrapment:
            work = exclude_entrapment(work)

        work = work.filter(
            pl.col(self.quantity_column) > 0,
            pl.col("Q.Value") <= self.qvalue_threshold,
            pl.col("Protein.Q.Value") <= self.protein_q_threshold,
            pl.col("PG.Q.Value") <= self.pg_q_threshold,
            pl.col("GG.Q.Value") <= self.gg_q_threshold,
        )

        work = work.with_columns(
            pl.col("Modified.Sequence").str.extract(r"\(([^()]*)\)", group_index=1).alias("first_modif"),
            pl.col("Modified.Sequence").str.find(r"\(").alias("first_modif_pos"),
        )

        work = work.with_columns(
            pl.col("first_modif").map_elements(self._map_nterm_modification, return_dtype=pl.Utf8).alias("mapped_modif")
        )

        work = work.with_columns(
            pl.when(pl.col("first_modif").is_null())
            .then(pl.lit("n"))
            .when(
                pl.col("first_modif")
                .str.contains(r"89\.030[0-9]")
                .fill_null(False)
                & (pl.col("first_modif_pos") == 0)
            )
            .then(pl.lit("2PCA"))
            .when((pl.col("first_modif_pos") == 0) & pl.col("mapped_modif").is_not_null())
            .then(pl.col("mapped_modif"))
            .otherwise(pl.lit("n"))
            .alias("nterm_modif")
        )

        work = work.with_columns(
            pl.when(pl.col("nterm_modif").str.contains("Dimethyl", literal=True, strict=False).fill_null(False))
            .then(pl.lit("Dimethyl"))
            .when(pl.col("nterm_modif").str.contains("TMT", literal=True, strict=False).fill_null(False))
            .then(pl.lit("TMT"))
            .otherwise(pl.col("nterm_modif"))
            .alias("nterm_modif"),
            (pl.col("nterm_modif") + pl.lit("_") + pl.col("Stripped.Sequence")).alias("nterm_modif_peptide"),
        )

        # DIA-NN represents shared peptides with semicolon-separated protein
        # accessions. Split these before the FASTA-based termini annotation;
        # keeping the whole group as one key makes every shared peptide look
        # unmapped even when all individual accessions exist in the FASTA.
        peptide2protein = self.expand_protein_ids(
            work,
            key_columns=("nterm_modif_peptide", "Stripped.Sequence"),
        ).rename({"Stripped.Sequence": "peptide"})

        matrix = self._summarize_to_matrix(work)
        if matrix.is_empty():
            return pl.DataFrame(
                {
                    "nterm_modif_peptide": pl.Series([], dtype=pl.Utf8),
                    "nterm_modif": pl.Series([], dtype=pl.Utf8),
                    "peptide": pl.Series([], dtype=pl.Utf8),
                    "protein": pl.Series([], dtype=pl.Utf8),
                }
            )

        matrix = matrix.with_columns(
            pl.col("nterm_modif_peptide").str.split_exact("_", 1).alias("split_parts")
        ).with_columns(
            pl.col("split_parts").struct.field("field_0").alias("nterm_modif"),
            pl.col("split_parts").struct.field("field_1").alias("peptide"),
        ).drop("split_parts")

        out = (
            matrix.join(peptide2protein, on="nterm_modif_peptide", how="left")
            .select(
                "nterm_modif_peptide",
                "nterm_modif",
                "peptide",
                "protein",
                *self._sorted_sample_columns(matrix),
            )
            .sort("peptide")
        )
        return out

    @staticmethod
    def _normalize_input_columns(df: pl.DataFrame) -> pl.DataFrame:
        if "File.Name" in df.columns:
            return df
        if "Run" in df.columns:
            return df.rename({"Run": "File.Name"})
        return df

    def _summarize_to_matrix(self, df: pl.DataFrame) -> pl.DataFrame:
        deduplicated = df.unique(
            subset=["File.Name", "nterm_modif_peptide", self.quantity_column],
            keep="first",
            maintain_order=False,
        )

        quantity = pl.col(self.quantity_column)
        if self.summarization == "SUM":
            aggregated = deduplicated.group_by(["nterm_modif_peptide", "File.Name"]).agg(
                quantity.sum().alias("abundance")
            )
        elif self.summarization == "MAX":
            aggregated = deduplicated.group_by(["nterm_modif_peptide", "File.Name"]).agg(
                quantity.max().alias("abundance")
            )
        else:
            raise ValueError(f"Unsupported summarization mode: {self.summarization}")

        wide = aggregated.pivot(index="nterm_modif_peptide", columns="File.Name", values="abundance")
        if wide.is_empty():
            return wide

        renamed = {}
        for col in wide.columns:
            if col == "nterm_modif_peptide":
                continue
            renamed[col] = self._clean_sample_name(col)
        wide = wide.rename(renamed)

        sample_cols = self._sample_columns(wide)
        wide = wide.with_columns(
            [
                # pl.when(pl.col(c) <= 0)
                pl.when((pl.col(c) <= 0) | pl.col(c).is_infinite() | pl.col(c).is_nan())
                .then(None)
                .otherwise(pl.col(c).log(base=2))
                .alias(c)
                for c in sample_cols
            ]
        )

        if not self.apply_mad_scaling:
            return wide

        long_df = wide.unpivot(
            index="nterm_modif_peptide",
            on=sample_cols,
            variable_name="sample",
            value_name="log2_ratio_2_ref",
        )

        long_df = (
            long_df.group_by(["nterm_modif_peptide", "sample"])
            .agg(pl.col("log2_ratio_2_ref").median().alias("log2_rat2ref_group"))
            .with_columns(
                pl.col("log2_rat2ref_group").median().over("sample").alias("Mi")
            )
            .with_columns(
                pl.col("Mi").median().alias("M0"),
                (pl.col("log2_rat2ref_group") - pl.col("Mi")).alias("RCij"),
            )
            .with_columns(
                pl.col("RCij").abs().median().over("sample").alias("MADi")
            )
            .with_columns(
                pl.col("MADi").median().alias("MAD0")
            )
            .with_columns(
                pl.when((pl.col("MADi") == 0) | pl.col("MADi").is_null())
                .then(None)
                .otherwise((pl.col("RCij") / pl.col("MADi")) * pl.col("MAD0") + pl.col("M0"))
                .alias("RNij")
            )
            .select("nterm_modif_peptide", "sample", "RNij")
        )

        return long_df.pivot(index="nterm_modif_peptide", columns="sample", values="RNij")

    @staticmethod
    def _sample_columns(df: pl.DataFrame) -> list[str]:
        return [c for c in df.columns if c not in {"nterm_modif_peptide", "nterm_modif", "peptide", "protein"}]

    @classmethod
    def _sorted_sample_columns(cls, df: pl.DataFrame) -> list[str]:
        def sort_key(col: str) -> tuple[int, str, int]:
            m = re.search(r"(\d+)$", col)
            if m:
                return (0, col[: m.start()], int(m.group(1)))
            return (1, col, 0)

        return sorted(cls._sample_columns(df), key=sort_key)

    def _apply_significant_digits(self, df: pl.DataFrame) -> pl.DataFrame:
        sample_cols = self._sample_columns(df)
        if not sample_cols:
            return df

        digits = max(int(self.significant_digits), 1)
        fmt = f"{{:.{digits}g}}"

        return df.with_columns(
            [
                pl.col(c)
                .cast(pl.Float64, strict=False)
                .map_elements(
                    lambda x: None if x is None else float(fmt.format(x)),
                    return_dtype=pl.Float64,
                )
                .alias(c)
                for c in sample_cols
            ]
        )

    @staticmethod
    def _clean_sample_name(value: str) -> str:
        sample = value.replace("\\", "/").split("/")[-1]
        if "." in sample:
            sample = sample.split(".", maxsplit=1)[0]
        return sample

    def _map_nterm_modification(self, modif: str | None) -> str | None:
        if modif is None:
            return None

        token = modif.strip()
        for key, name in self.unimod_mapping.items():
            if key in token:
                return name

        # Legacy DIA-NN mass encodings explicitly handled by the source QMD's
        # modified adapter for N-terminal pyroglutamate-like products.
        if any(
            re.search(pattern, token)
            for pattern in (
                r"76\.968[0-9]*",
                r"111\.027[0-9]*",
                r"108\.941[0-9]*",
                r"78\.975[0-9]*",
                r"110\.947[0-9]*",
                r"-51\.089[0-9]*",
            )
        ):
            return "Pyro-Glu"
        if re.search(r"304\.207[0-9]", token) or re.search(r"229\.162[0-9]", token):
            return "TMT"
        if re.search(r"42\.010[0-9]", token):
            return "Acetyl"
        if re.search(r"28\.031[0-9]", token) or re.search(r"34\.063[0-9]", token) or re.search(r"36\.075[0-9]", token):
            return "Dimethyl"
        if re.search(r"-17\.026[0-9]", token) or re.search(r"-18\.010[0-9]", token):
            return "Pyro-Glu"

        return None
