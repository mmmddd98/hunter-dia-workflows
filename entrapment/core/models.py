from dataclasses import dataclass


@dataclass(frozen=True)
class Protein:
    accession: str
    sequence: str
    description: str
    organism: str
    gene: str

    def __str__(self) -> str:
        return f">{self.accession}\n{self.sequence[:10]}..."
