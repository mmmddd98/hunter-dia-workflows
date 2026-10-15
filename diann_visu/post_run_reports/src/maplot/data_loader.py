from pathlib import Path

import polars as pl


class ResultDataLoader:
    """Load and write DIA-NN result tables in common tabular formats."""

    def __init__(self, input_file: str | Path, qvalue_threshold: float = 0.01) -> None:
        """
        Configure file loading for one result table.

        Args:
            input_file: Path to a parquet/tsv/txt/csv result table.
            qvalue_threshold: Max allowed `Q.Value` when that column exists.
        """
        self.input_file = Path(input_file).expanduser().resolve()
        self.qvalue_threshold = qvalue_threshold

    def load(self) -> pl.DataFrame:
        """Load the configured table and filter by `Q.Value` when present."""
        df = self.load_table(self.input_file)

        if "Q.Value" in df.columns:
            df = df.filter(pl.col("Q.Value") <= self.qvalue_threshold)
        return df

    @staticmethod
    def load_table(path: str | Path) -> pl.DataFrame:
        """Load a parquet, CSV, or TSV table based on the file suffix."""
        table_path = Path(path).expanduser().resolve()
        suffix = table_path.suffix.lower()
        if suffix == ".parquet":
            return pl.read_parquet(table_path)
        if suffix in {".tsv", ".txt"}:
            return pl.read_csv(table_path, separator="\t", null_values=["", "NA", "NaN"])
        if suffix == ".csv":
            return pl.read_csv(table_path, null_values=["", "NA", "NaN"])
        raise ValueError(f"Unsupported file type: {suffix}")

    @staticmethod
    def write_table(df: pl.DataFrame, path: str | Path) -> Path:
        """Write a table to parquet, CSV, or TSV and return the output path."""
        table_path = Path(path).expanduser().resolve()
        table_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = table_path.suffix.lower()
        if suffix == ".parquet":
            df.write_parquet(table_path)
        elif suffix in {".tsv", ".txt"}:
            df.write_csv(table_path, separator="\t", null_value="NA")
        elif suffix == ".csv":
            df.write_csv(table_path, null_value="NA")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
        return table_path
