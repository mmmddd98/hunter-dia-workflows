from pathlib import Path
from typing import Optional, overload

from .fasta_io import FastaReader, FastaWriter
from .models import Protein
from .shuffling import ProteinShuffler, ShuffleMode


class EntrapmentGenerator:
    def __init__(
        self,
        reader: Optional[FastaReader] = None,
        writer: Optional[FastaWriter] = None,
    ) -> None:
        self.reader = reader or FastaReader()
        self.writer = writer or FastaWriter()

    @overload
    def entrap(
        self,
        fasta_path: Path | str,
        list_of_proteins: None = ...,
    ) -> tuple[list[Protein], Path]: ...

    @overload
    def entrap(
        self,
        fasta_path: None = ...,
        list_of_proteins: list[Protein] = ...,
    ) -> tuple[list[Protein], Path]: ...

    def entrap(
        self,
        fasta_path: Optional[Path | str] = None,
        list_of_proteins: Optional[list[Protein]] = None,
        out_dir: Optional[Path | str] = None,
        filename: str = "entrapment.fasta",
        seed: int = 1337,
        keep_first_letter: bool = True,
        shuffle_mode: ShuffleMode = "fixed-r",
        combine_with_original: bool = True,
    ) -> tuple[list[Protein], Path]:
        """Create shuffled entrapment proteins and write a FASTA file.

        If combine_with_original is True, the written FASTA contains the original
        proteins first and the shuffled entrapment proteins after them.
        """
        if (fasta_path is None) == (list_of_proteins is None):
            raise ValueError("Provide exactly one of fasta_path OR list_of_proteins")

        proteins = (
            list_of_proteins
            if list_of_proteins is not None
            else self.reader.parse(Path(fasta_path))
        )

        shuffler = ProteinShuffler(seed=seed)
        shuffled = [
            shuffler.shuffle(
                protein,
                keep_first_letter=keep_first_letter,
                mode=shuffle_mode,
            )
            for protein in proteins
        ]

        if out_dir is None:
            out_dir = (
                Path(fasta_path).parent / "entrapment_out"
                if fasta_path is not None
                else Path("entrapment_out")
            )

        proteins_to_write = [*proteins, *shuffled] if combine_with_original else shuffled
        out_path = self.writer.write(proteins_to_write, out_dir=out_dir, filename=filename)
        return shuffled, out_path
