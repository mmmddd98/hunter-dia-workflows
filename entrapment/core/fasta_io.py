from pathlib import Path
import re
from typing import Optional

from .models import Protein


class FastaReader:
    def parse(self, path: Path | str) -> list[Protein]:
        path = Path(path)
        proteins: list[Protein] = []
        pattern = (
            r"^(?P<database>\w+)\|"
            r"(?P<accession>[^|]+)\|"
            r"(?P<entry_name>\S+)\s+"
            r"(?P<description>.+?)\s*"
            r"OS=(?P<organism>.+?)\s+"
            r"OX=(?P<taxonomy_id>\d+)\s+"
            r"(?:GN=(?P<gene_name>.+?)\s+)?"
            r"PE=(?P<protein_existence>\d+)\s+"
            r"SV=(?P<sequence_version>\d+)"
        )

        for entry in path.read_text().split(">")[1:]:
            lines = entry.split("\n")
            header = lines[0]
            sequence = "".join(lines[1:])
            match = re.match(pattern=pattern, string=header)

            if match:
                info = match.groupdict()
                proteins.append(
                    Protein(
                        accession=info["accession"],
                        sequence=sequence,
                        description=info["description"],
                        organism=info["organism"],
                        gene=info.get("gene_name") or "",
                    )
                )
            else:
                print(f"Could not parse header: {header}")

        return proteins


class FastaWriter:
    def __init__(self) -> None:
        self.out_dir: Optional[Path] = None

    def write(
        self,
        proteins: list[Protein],
        out_dir: Path | str,
        filename: str = "entrapment.fasta",
        include_fields: tuple[str, ...] = ("description", "organism", "gene"),
        line_width: int = 60,
        overwrite: bool = True,
    ) -> Path:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / filename
        mode = "w" if overwrite else "x"

        with path.open(mode) as fh:
            for protein in proteins:
                fh.write(self._format_header(protein, include_fields) + "\n")
                self._write_sequence(fh, protein.sequence, line_width)

        return path

    def _format_header(
        self,
        protein: Protein,
        include_fields: tuple[str, ...],
    ) -> str:
        header = f">{protein.accession}"
        parts = []

        if "description" in include_fields and protein.description:
            parts.append(protein.description)
        if "organism" in include_fields and protein.organism:
            parts.append(f"OS={protein.organism}")
        if "gene" in include_fields and protein.gene:
            parts.append(f"GN={protein.gene}")

        if parts:
            header = header + " " + " ".join(parts)

        return header

    def _write_sequence(self, fh, sequence: str, line_width: int) -> None:
        if line_width and line_width > 0:
            for i in range(0, len(sequence), line_width):
                fh.write(sequence[i : i + line_width] + "\n")
        else:
            fh.write(sequence + "\n")
