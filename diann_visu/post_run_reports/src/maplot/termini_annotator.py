from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl


SUPPORTED_ORGANISMS = [
    "mouse",
    "human",
    "medicago_trucantula",
    "medicago_truncatula",
    "rhizobium_melitoli",
    "pig",
    "human_iso",
    "arabidopsis",
    "ecoli",
    "rat",
    "yeast",
    "c_elegans",
    "triplemix"
]

PROCESSING_FEATURE_TYPES = ["INIT_MET", "PROPEP", "SIGNAL", "TRANSIT", "CHAIN"]
EXPECTED_NTERM_MODIFICATIONS = {
    "TMT",
    "Dimethyl",
    "Acetyl",
    "2PCA",
}


def fasta_accession_from_header(header: str) -> str | None:
    pipe_match = re.search(r"\|([A-Z0-9_]+)\|", header)
    if pipe_match:
        return pipe_match.group(1)
    first_token = header[1:].strip().split(maxsplit=1)[0] if header.startswith(">") else ""
    return first_token or None


def fasta_organism_from_header(header: str) -> str | None:
    """Extract and normalize the UniProt ``OS=`` organism annotation."""
    match = re.search(r"\bOS=(.+?)(?=\s(?:OX|GN|PE|SV)=|$)", header)
    if not match:
        return None
    organism = match.group(1).strip()
    if organism.startswith("Escherichia coli"):
        return "Escherichia coli"
    return organism or None


@dataclass
class TerminiAnnotator:
    """
    Annotate peptides with cleavage site information, UniProt processing features,
    and predicted protease activity from MEROPS.

    Attributes:
        organism: Organism key used for bundled reference tables.
        sense: Cleavage sense ("C" or "N") for specificity logic.
        specificity: Cleavage residues as a pipe-separated list (default `"K|R"`).
            Use one-letter amino-acid tokens, for example `"K|R"` (trypsin-like),
            `"D|E"` (acidic), or `"K"` (single residue).
        n_residues_area: Window size around cleavage for sequence context columns.
        distinct: Keep distinct annotation rows in helper joins.
        pssm_prediction: Enable MEROPS PSSM-based protease prediction.
        all_occurrences: Map every exact peptide occurrence (including overlaps)
            in each listed FASTA protein instead of only the first occurrence.
        met_excision: Treat a peptide at position 2 after an initiator M as a
            specific protein N terminus.
        pssm_protease_ids: Optional subset of protease IDs to score.
        pssm_top_n: Number of top-scoring proteases to keep.
        uniprot_processing: Optional preloaded UniProt processing table.
        merops_sites: Optional preloaded known MEROPS cleavage sites.
        merops_accession_to_protease: Optional accession-to-protease lookup table.

    Example (trypsin-like setup):
        >>> ta = TerminiAnnotator(
        ...     organism="human",
        ...     sense="C",
        ...     specificity="K|R",
        ...     n_residues_area=10,
        ... )
    """

    organism: str = "human"
    sense: Literal["C", "N"] = "C"
    specificity: str = "K|R"
    n_residues_area: int = 10
    distinct: bool = True
    pssm_prediction: bool = True
    all_occurrences: bool = True
    met_excision: bool = False
    pssm_protease_ids: list[str] | None = None
    pssm_top_n: int = 5
    uniprot_processing: pl.DataFrame | None = None
    merops_sites: pl.DataFrame | None = None
    merops_accession_to_protease: pl.DataFrame | None = None

    def __post_init__(self):
        if self.organism not in SUPPORTED_ORGANISMS:
            raise ValueError(
                f"Organism '{self.organism}' not supported. "
                f"Choose from: {', '.join(SUPPORTED_ORGANISMS)}"
            )
        if self.sense not in {"C", "N"}:
            raise ValueError("sense must be 'C' or 'N'")
        spec_tokens = [token.strip() for token in self.specificity.split("|")]
        if not all(spec_tokens):
            raise ValueError("specificity must be a non-empty pipe-separated list like 'K|R'")
        if not all(re.fullmatch(r"[A-Z]", token) for token in spec_tokens):
            raise ValueError("specificity tokens must be single uppercase amino-acid letters, e.g. 'K|R'")
        self.specificity = "|".join(spec_tokens)
        if self.pssm_top_n < 1:
            raise ValueError("pssm_top_n must be >= 1")

    def annotate(
        self,
        peptides_df: pl.DataFrame,
        fasta_path: str | Path,
    ) -> pl.DataFrame:
        """
        Main annotation pipeline.

        Parameters
        ----------
        peptides_df : pl.DataFrame
            DataFrame from DiannAdapter with columns:
            nterm_modif_peptide, nterm_modif, peptide, protein, and sample columns
        fasta_path : str | Path
            Path to UniProt FASTA file

        Returns
        -------
        pl.DataFrame
            Annotated DataFrame with cleavage site information
        """
        fasta_path = Path(fasta_path)
        if not fasta_path.exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

        required_cols = {"peptide", "protein", "nterm_modif"}
        missing = required_cols.difference(peptides_df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        fasta_df = self._load_fasta(fasta_path)
        peptide2protein = peptides_df.select(["peptide", "protein", "nterm_modif"]).unique(maintain_order=True)

        prot2pept2fasta = self._map_peptides_to_protein(peptide2protein, fasta_df)
        annotated_basic = self._annotate_core(prot2pept2fasta)
        canonical_annotations = self._annotate_uniprot_processing(annotated_basic)

        annotated_with_quant = peptides_df.join(
            annotated_basic.rename({"start_position": "peptide_start", "end_position": "peptide_end"}),
            on=["peptide", "protein", "nterm_modif"],
            how="left",
        ).join(
            canonical_annotations,
            on=["peptide", "protein", "peptide_start", "p1_position", "cleavage_sequence"],
            how="left",
            nulls_equal=True,
        ).with_columns(
            pl.when(pl.col("uniprot_processing_type").is_null())
            .then(pl.lit("not_canonical_no_procc_annot"))
            .otherwise(pl.col("uniprot_processing_type"))
            .alias("uniprot_processing_type")
        )

        annotated_with_merops = self._add_known_merops_sites(annotated_with_quant)
        if self.pssm_prediction:
            annotated_with_merops = self._add_merops_annotations(annotated_with_merops)
        else:
            eligible = (pl.col("nterm_modif") != "n") | pl.col("specificity").is_in(["semi_Nterm", "semi_Cterm"])
            annotated_with_merops = annotated_with_merops.with_columns(
                (
                    pl.col("protease_merops_ids")
                    if "protease_merops_ids" in annotated_with_merops.columns
                    else pl.lit(None, dtype=pl.Utf8).alias("protease_merops_ids")
                ),
                (
                    pl.col("protease_merops_names")
                    if "protease_merops_names" in annotated_with_merops.columns
                    else pl.lit(None, dtype=pl.Utf8).alias("protease_merops_names")
                ),
                pl.when(eligible).then(pl.lit("")).otherwise(None).alias("predicted_protease_activity_ids"),
                pl.when(eligible).then(pl.lit("")).otherwise(None).alias("predicted_protease_activity_names"),
                pl.when(eligible).then(pl.lit("")).otherwise(None).alias("predicted_protease_activity"),
            )

        return self._relocate_output_columns(annotated_with_merops)

    def _load_fasta(self, fasta_path: str | Path) -> pl.DataFrame:
        """Load and parse UniProt FASTA file."""
        fasta_path = Path(fasta_path).expanduser().resolve()
        fasta_records: dict[str, tuple[str, str | None]] = {}

        def store_record(
            protein_id: str | None,
            sequence_parts: list[str],
            protein_organism: str | None,
        ) -> None:
            if protein_id is None:
                return
            sequence = "".join(sequence_parts)
            previous = fasta_records.get(protein_id)
            if previous is None:
                fasta_records[protein_id] = (sequence, protein_organism)
                return
            if previous[0] == sequence:
                return
            # Entrapment FASTA bundles can contain the same ENT accession
            # more than once with independently shuffled sequences. The
            # accession is ambiguous in that case, but aborting the complete
            # dataset would prevent all termini plots. Keep the first record
            # deterministically; non-entrapment collisions remain strict.
            if protein_id.startswith("ENT_"):
                return
            raise ValueError(
                f"Duplicated FASTA accession detected with different sequences: {protein_id}"
            )

        with open(fasta_path, "r") as f:
            current_id = None
            current_seq = []
            current_organism = None

            for line in f:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    store_record(current_id, current_seq, current_organism)
                    current_id = fasta_accession_from_header(line)
                    current_organism = fasta_organism_from_header(line)
                    current_seq = []
                else:
                    if current_id is not None:
                        current_seq.append(line)

            store_record(current_id, current_seq, current_organism)

        if len(fasta_records) == 0:
            raise ValueError("No sequences parsed from FASTA file")

        df = pl.DataFrame(
            {
                "protein": list(fasta_records.keys()),
                "protein_sequence": [record[0] for record in fasta_records.values()],
                "protein_organism": [record[1] for record in fasta_records.values()],
            }
        )

        return df

    def _map_peptides_to_protein(
        self, peptide2protein: pl.DataFrame, fasta_df: pl.DataFrame
    ) -> pl.DataFrame:
        """Find every peptide occurrence in each mapped protein sequence."""
        rows = []
        for row in peptide2protein.join(fasta_df, on="protein", how="left").iter_rows(named=True):
            peptide = row["peptide"]
            protein_sequence = row["protein_sequence"]
            if protein_sequence is None:
                positions = [None]
            else:
                positions = []
                search_from = 0
                while True:
                    zero_based = protein_sequence.find(peptide, search_from)
                    if zero_based < 0:
                        break
                    positions.append(zero_based)
                    if not self.all_occurrences:
                        break
                    search_from = zero_based + 1  # include overlapping occurrences
                if not positions:
                    positions = [None]

            for zero_based in positions:
                if zero_based is None:
                    start_position = end_position = aa_before = aa_after = protein_length = None
                else:
                    start_position = zero_based + 1  # R-compatible one-based coordinates
                    end_position = zero_based + len(peptide)
                    aa_before = protein_sequence[zero_based - 1] if zero_based > 0 else None
                    aa_after = protein_sequence[end_position] if end_position < len(protein_sequence) else None
                    protein_length = len(protein_sequence)

                rows.append(
                    {
                        **row,
                        "start_position": start_position,
                        "end_position": end_position,
                        "last_aa": peptide[-1:] or None,
                        "first_aa": peptide[:1] or None,
                        "aa_after": aa_after,
                        "aa_before": aa_before,
                        "sense": self.sense,
                        "protein_length": protein_length,
                    }
                )

        return pl.DataFrame(rows) if rows else pl.DataFrame()

    def _annotate_core(self, df: pl.DataFrame) -> pl.DataFrame:
        """Annotate cleavage specificity and sequence windows using R-compatible coordinates."""
        spec_parts = self.specificity.split("|")
        spec_pattern = "|".join(re.escape(aa) for aa in spec_parts)

        def matches(value: str | None) -> bool:
            return bool(value is not None and re.search(spec_pattern, value))

        def check_specificity(row: dict) -> str | None:
            sense = row["sense"]
            last_aa = row["last_aa"]
            aa_before = row["aa_before"]
            first_aa = row["first_aa"]
            aa_after = row["aa_after"]
            start_position = row.get("start_position")
            end_position = row.get("end_position")
            protein_length = row.get("protein_length")
            if start_position is None or end_position is None or protein_length is None:
                return None

            # A protein boundary is a specific terminus. For C-terminal
            # cleavage (the Results 4.5 R* setting), the residue before the
            # peptide defines N-terminal cleavage and the peptide's last
            # residue defines C-terminal cleavage. The N-sense mode uses the
            # corresponding outward-facing residues.
            if sense == "C":
                n_specific = start_position == 1 or matches(aa_before)
                c_specific = end_position == protein_length or matches(last_aa)
            else:
                n_specific = start_position == 1 or matches(first_aa)
                c_specific = end_position == protein_length or matches(aa_after)
            n_specific = n_specific or (
                self.met_excision and start_position == 2
                and row.get("protein_sequence", "").startswith("M")
            )

            if n_specific and c_specific:
                return "specific"
            if n_specific:
                return "semi_Cterm"
            if c_specific:
                return "semi_Nterm"
            return "unspecific"

        rows = []
        for row in df.iter_rows(named=True):
            specificity = check_specificity(row)
            annotated = {**row, "specificity": specificity}
            annotated.update(self._sequence_window_annotations(annotated))
            rows.append(annotated)

        return pl.DataFrame(rows) if rows else df

    def _sequence_window_annotations(self, row: dict) -> dict:
        protein_sequence = row.get("protein_sequence")
        start_position = row.get("start_position")
        end_position = row.get("end_position")
        specificity = row.get("specificity")

        empty = {
            "five_res_before": None,
            "five_res_after": None,
            "x_res_before": None,
            "x_res_after": None,
            "cleavage_site": "NA | NA",
            "cleavage_sequence": "NANA",
            "cleavage_sequence_x": "NANA",
            "p1_position": None,
            "p1_prime_position": None,
            "p1_residue": None,
            "p1_prime_residue": row.get("first_aa") if row.get("sense") == "C" else row.get("aa_after"),
            "p1_position_percentage": None,
            "met_clipping": False,
        }
        if protein_sequence is None or start_position is None or end_position is None or specificity is None:
            return empty

        nterm_modif = row.get("nterm_modif")
        expected_modification = nterm_modif in EXPECTED_NTERM_MODIFICATIONS
        del expected_modification  # Kept explicit to mirror the R branches; both paths are equivalent.

        five_res_before = self._slice_before(protein_sequence, start_position, end_position, specificity, 5).rjust(5, "X")
        five_res_after = self._slice_after(protein_sequence, start_position, end_position, specificity, 5).ljust(5, "X")
        x_res_before = self._slice_before(
            protein_sequence, start_position, end_position, specificity, self.n_residues_area
        ).rjust(self.n_residues_area, "X")
        x_res_after = self._slice_after(
            protein_sequence, start_position, end_position, specificity, self.n_residues_area
        ).ljust(self.n_residues_area, "X")

        if specificity == "semi_Nterm":
            p1_position = start_position - 1
            p1_prime_position = start_position
        elif specificity == "semi_Cterm":
            p1_position = end_position
            p1_prime_position = end_position + 1
        else:
            p1_position = start_position - 1
            p1_prime_position = start_position

        if p1_position == 0:
            p1_residue = "N-term"
        elif specificity != "specific":
            p1_residue = five_res_before[-1]
        elif row["sense"] == "C" and row.get("aa_before") is not None:
            p1_residue = row["aa_before"]
        elif row["sense"] == "N" and row.get("last_aa") is not None:
            p1_residue = row["last_aa"]
        else:
            p1_residue = None

        if p1_prime_position == row.get("protein_length"):
            p1_prime_residue = "C-term"
        elif specificity != "specific":
            p1_prime_residue = five_res_after[0]
        elif row["sense"] == "C" and row.get("first_aa") is not None:
            p1_prime_residue = row["first_aa"]
        elif row["sense"] == "N" and row.get("aa_after") is not None:
            p1_prime_residue = row["aa_after"]
        else:
            p1_prime_residue = None

        # Initiator-M clipping is a separate processing annotation, not an
        # arbitrary internal-M cleavage rule. Coordinates are one-based.
        met_clipping = (
            self.met_excision and start_position == 2 and protein_sequence.startswith("M")
        )

        return {
            "five_res_before": five_res_before,
            "five_res_after": five_res_after,
            "x_res_before": x_res_before,
            "x_res_after": x_res_after,
            "cleavage_site": f"{five_res_before[1:5]} | {five_res_after[0:4]}",
            "cleavage_sequence": f"{five_res_before[1:5]}{five_res_after[0:4]}",
            "cleavage_sequence_x": f"{x_res_before}{x_res_after}",
            "p1_position": p1_position,
            "p1_prime_position": p1_prime_position,
            "p1_residue": p1_residue,
            "p1_prime_residue": p1_prime_residue,
            "p1_position_percentage": p1_position / len(protein_sequence) * 100,
            "met_clipping": met_clipping,
        }

    def _annotate_uniprot_processing(self, annotated_basic: pl.DataFrame) -> pl.DataFrame:
        base_cols = [
            "peptide",
            "protein",
            "peptide_start",
            "p1_position",
            "matches_p1_prime",
            "cleavage_sequence",
            "uniprot_processing_type",
            "processing_annotation_start",
            "processing_annotation_end",
        ]
        if annotated_basic.is_empty():
            return pl.DataFrame({col: [] for col in base_cols})

        candidates = annotated_basic.rename({"start_position": "peptide_start", "end_position": "peptide_end"}).filter(
            (~pl.col("protein").str.contains("Biognosys", literal=True))
            & ((pl.col("nterm_modif") != "n") | (pl.col("specificity").is_in(["semi_Nterm", "semi_Cterm"])))
        )
        if candidates.is_empty():
            return pl.DataFrame({col: [] for col in base_cols})

        if self.uniprot_processing is None:
            return candidates.with_columns(
                pl.lit(False).alias("matches_p1_prime"),
                pl.lit("not_canonical").alias("uniprot_processing_type"),
                pl.lit(None, dtype=pl.Int64).alias("processing_annotation_start"),
                pl.lit(None, dtype=pl.Int64).alias("processing_annotation_end"),
            ).with_columns(
                pl.when((pl.col("p1_residue") == "M") & (pl.col("p1_position") < 3))
                .then(pl.lit("INIT_MET_not_canonical"))
                .when(pl.col("p1_prime_position") == 1)
                .then(pl.lit("Intact_ORF"))
                .otherwise(pl.col("uniprot_processing_type"))
                .alias("uniprot_processing_type")
            ).select(
                "peptide",
                "protein",
                "peptide_start",
                "p1_position",
                "matches_p1_prime",
                "cleavage_sequence",
                "uniprot_processing_type",
                "processing_annotation_start",
                "processing_annotation_end",
            )

        processing = self.uniprot_processing
        rename_map = {}
        if "accession" in processing.columns and "protein" not in processing.columns:
            rename_map["accession"] = "protein"
        if "begin" in processing.columns and "processing_annotation_start" not in processing.columns:
            rename_map["begin"] = "processing_annotation_start"
        if "end" in processing.columns and "processing_annotation_end" not in processing.columns:
            rename_map["end"] = "processing_annotation_end"
        if rename_map:
            processing = processing.rename(rename_map)

        processing = processing.filter(pl.col("type").is_in(PROCESSING_FEATURE_TYPES)).select(
            "protein",
            "type",
            pl.col("processing_annotation_start").cast(pl.Int64, strict=False),
            pl.col("processing_annotation_end").cast(pl.Int64, strict=False),
        )
        joined = candidates.join(processing, on="protein", how="left")
        joined = joined.with_columns(
            (pl.col("p1_prime_position") - pl.col("processing_annotation_end"))
            .abs()
            .le(4)
            .fill_null(False)
            .alias("matches_p1_prime")
        ).with_columns(
            pl.when(pl.col("matches_p1_prime") & (pl.col("type") != "CHAIN"))
            .then(pl.col("type"))
            .when(pl.col("matches_p1_prime") & (pl.col("type") == "CHAIN"))
            .then(pl.lit("CHAIN"))
            .otherwise(pl.lit("not_canonical"))
            .alias("uniprot_processing_type")
        ).filter(pl.col("uniprot_processing_type") != "CHAIN")

        if self.distinct:
            matched = joined.filter(~pl.col("uniprot_processing_type").is_in(["not_canonical", "not_canonical_no_procc_annot"]))
            matched = matched.unique(subset=["peptide", "protein", "matches_p1_prime"], keep="first", maintain_order=True)
            unmatched = joined.filter(~pl.col("peptide").is_in(matched.get_column("peptide"))).unique(
                subset=["peptide", "protein", "matches_p1_prime", "uniprot_processing_type"],
                keep="first",
                maintain_order=True,
            )
            joined = pl.concat([matched, unmatched], how="vertical_relaxed")

        joined = joined.with_columns(
            pl.when(
                pl.col("uniprot_processing_type").is_in(["not_canonical", "not_canonical_no_procc_annot"])
                & (pl.col("p1_residue") == "M")
                & (pl.col("p1_position") < 3)
            )
            .then(pl.lit("INIT_MET_not_canonical"))
            .when(
                pl.col("uniprot_processing_type").is_in(["not_canonical", "not_canonical_no_procc_annot"])
                & (pl.col("p1_prime_position") == 1)
            )
            .then(pl.lit("Intact_ORF"))
            .otherwise(pl.col("uniprot_processing_type"))
            .alias("uniprot_processing_type")
        )

        return joined.select(base_cols)

    def _add_merops_annotations(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add MEROPS-based annotations (known sites + PSSM predictions).

        Note: Currently returns placeholders. Full implementation requires
        MEROPS data tables (merops_sites, merops_pssm, merops_accession_to_protease).
        """
        result = df
        if self.merops_sites is not None:
            result = self._add_known_merops_sites(result)

        result = result.with_columns(
            pl.lit("").alias("protease_merops_ids"),
            pl.lit("").alias("protease_merops_names"),
            pl.lit("").alias("predicted_protease_activity_ids"),
            pl.lit("").alias("predicted_protease_activity_names"),
            pl.lit("").alias("predicted_protease_activity"),
        ) if "protease_merops_ids" not in result.columns else result

        return result

    def _add_known_merops_sites(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.merops_sites is None or self.merops_sites.is_empty():
            return df

        sites = self.merops_sites
        if not {"uniprot_acc", "position", "protease_id"}.issubset(sites.columns):
            return df

        names = self.merops_accession_to_protease
        site_map = sites.select(
            pl.concat_str([pl.col("uniprot_acc"), pl.col("position").cast(pl.Utf8)], separator="_").alias("site_key"),
            pl.col("protease_id").cast(pl.Utf8),
        ).unique()
        if names is not None and {"protease_id", "protease_name"}.issubset(names.columns):
            site_map = site_map.join(names.select("protease_id", "protease_name"), on="protease_id", how="left")
        else:
            site_map = site_map.with_columns(pl.col("protease_id").alias("protease_name"))

        site_map = self._summarize_merops_site_map(site_map)

        eligible = (pl.col("nterm_modif") != "n") | pl.col("specificity").is_in(["semi_Nterm", "semi_Cterm"])

        return df.with_columns(
            pl.concat_str(["protein", pl.col("p1_position").cast(pl.Utf8)], separator="_").alias("site_key")
        ).join(site_map, on="site_key", how="left").with_columns(
            pl.when(eligible)
            .then(pl.col("protease_merops_ids").fill_null(""))
            .otherwise(None)
            .alias("protease_merops_ids"),
            pl.when(eligible)
            .then(pl.col("protease_merops_names").fill_null(""))
            .otherwise(None)
            .alias("protease_merops_names"),
        ).drop("site_key")

    @staticmethod
    def _summarize_merops_site_map(site_map: pl.DataFrame) -> pl.DataFrame:
        grouped: dict[str, dict[str, set[str]]] = {}
        for row in site_map.iter_rows(named=True):
            site_key = row["site_key"]
            protease_id = row["protease_id"]
            protease_name = row["protease_name"] or protease_id
            values = grouped.setdefault(site_key, {"ids": set(), "names": set()})
            values["ids"].add(protease_id)
            values["names"].add(protease_name)

        return pl.DataFrame(
            {
                "site_key": site_key,
                "protease_merops_ids": "|".join(sorted(values["ids"])),
                "protease_merops_names": "|".join(sorted(values["names"], key=str.casefold)),
            }
            for site_key, values in grouped.items()
        )

    def _relocate_output_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        preferred = [
            "nterm_modif_peptide",
            "nterm_modif",
            "peptide",
            "protein",
            "protein_organism",
            "protein_length",
            "peptide_start",
            "peptide_end",
            "last_aa",
            "first_aa",
            "aa_after",
            "aa_before",
            "sense",
            "specificity",
            "five_res_before",
            "five_res_after",
            "x_res_before",
            "x_res_after",
            "cleavage_site",
            "cleavage_sequence",
            "cleavage_sequence_x",
            "p1_position",
            "p1_prime_position",
            "p1_residue",
            "p1_prime_residue",
            "p1_position_percentage",
            "met_clipping",
            "uniprot_processing_type",
            "processing_annotation_start",
            "processing_annotation_end",
            "matches_p1_prime",
            "protease_merops_ids",
            "protease_merops_names",
            "predicted_protease_activity_ids",
            "predicted_protease_activity_names",
            "predicted_protease_activity",
            "protein_sequence",
        ]
        existing = [col for col in preferred if col in df.columns]
        rest = [col for col in df.columns if col not in existing]
        return df.select(existing + rest)

    @staticmethod
    def _slice_before(
        protein_seq: str,
        start_pos: int,
        end_pos: int,
        specificity: str,
        width: int,
    ) -> str:
        if specificity == "semi_Cterm":
            return TerminiAnnotator._r_substr(protein_seq, end_pos - (width - 1), end_pos)
        return TerminiAnnotator._r_substr(protein_seq, 1 if start_pos - width <= 1 else start_pos - width, start_pos - 1)

    @staticmethod
    def _slice_after(
        protein_seq: str,
        start_pos: int,
        end_pos: int,
        specificity: str,
        width: int,
    ) -> str:
        protein_length = len(protein_seq)
        if specificity == "semi_Cterm":
            return TerminiAnnotator._r_substr(
                protein_seq,
                end_pos + 1,
                protein_length if end_pos + width >= protein_length else end_pos + width,
            )
        return TerminiAnnotator._r_substr(
            protein_seq,
            start_pos,
            protein_length if start_pos + (width - 1) >= protein_length else start_pos + (width - 1),
        )

    @staticmethod
    def _r_substr(value: str, start: int, stop: int) -> str:
        """R stringr::str_sub-like helper for positive 1-based inclusive bounds."""
        if start < 0:
            start = len(value) + start + 1
        if stop < 0:
            stop = len(value) + stop + 1
        if stop < 1 or start > len(value) or start > stop:
            return ""
        start = max(start, 1)
        stop = min(stop, len(value))
        return value[start - 1 : stop]

    @staticmethod
    def _extract_five_residues(
        protein_seq: str,
        start_pos: int,
        end_pos: int,
        specificity: str,
        direction: str,
        pad_char: str = "X",
    ) -> str:
        """
        Extract 5 residues before or after cleavage site.
        direction: 'before' or 'after'
        """
        seq_len = len(protein_seq)

        if direction == "before":
            if specificity == "semi_Nterm":
                before_start = max(0, start_pos - 5)
                snippet = protein_seq[before_start : start_pos]
            elif specificity == "semi_Cterm":
                before_start = max(0, end_pos - 5)
                snippet = protein_seq[before_start : end_pos]
            else:  # specific
                before_start = max(0, start_pos - 5)
                snippet = protein_seq[before_start : start_pos]

            snippet = snippet.rjust(5, pad_char)
        else:  # after
            if specificity == "semi_Nterm":
                after_end = min(seq_len, start_pos + 5)
                snippet = protein_seq[start_pos : after_end]
            elif specificity == "semi_Cterm":
                after_end = min(seq_len, end_pos + 5)
                snippet = protein_seq[end_pos : after_end]
            else:  # specific
                after_end = min(seq_len, start_pos + 5)
                snippet = protein_seq[start_pos : after_end]

            snippet = snippet.ljust(5, pad_char)

        return snippet
