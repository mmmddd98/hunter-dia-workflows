from dataclasses import replace
import random
from typing import Literal

from .models import Protein

ShuffleMode = Literal["fixed-r", "simple"]


class ProteinShuffler:
    def __init__(self, seed: int = 1337) -> None:
        self.seed = seed
        self.rng = random.Random(self.seed)

    def shuffle(
        self,
        protein: Protein,
        term: str = "ENT",
        keep_first_letter: bool = True,
        mode: ShuffleMode = "fixed-r",
    ) -> Protein:
        return replace(
            protein,
            accession=f"{term}_{protein.accession}",
            sequence=self._shuffle_seq(
                protein.sequence,
                keep_first_letter=keep_first_letter,
                mode=mode,
            ),
        )

    def _shuffle_seq(
        self,
        seq: str,
        keep_first_letter: bool = True,
        mode: ShuffleMode = "fixed-r",
    ) -> str:
        if mode == "fixed-r":
            return self._shuffle_seq_fixed_residue(
                seq,
                fixed_residue="R",
                keep_first_letter=keep_first_letter,
            )

        if mode == "simple":
            return self._shuffle_seq_simple(seq, keep_first_letter=keep_first_letter)

        raise ValueError(f"Unsupported shuffle mode: {mode}")

    def _shuffle_seq_simple(self, seq: str, keep_first_letter: bool = True) -> str:
        if keep_first_letter:
            to_shuffle = list(seq[1:])
            self.rng.shuffle(to_shuffle)
            return seq[0] + "".join(to_shuffle)

        to_shuffle = list(seq)
        self.rng.shuffle(to_shuffle)
        return "".join(to_shuffle)

    def _shuffle_seq_fixed_residue(
        self,
        seq: str,
        fixed_residue: str,
        keep_first_letter: bool = True,
    ) -> str:
        fixed_positions = {
            idx for idx, amino_acid in enumerate(seq) if amino_acid == fixed_residue
        }
        if keep_first_letter and seq:
            fixed_positions.add(0)

        shuffle_positions = [
            idx for idx in range(len(seq)) if idx not in fixed_positions
        ]
        shuffled_letters = [seq[idx] for idx in shuffle_positions]
        self.rng.shuffle(shuffled_letters)

        out = list(seq)
        for idx, amino_acid in zip(shuffle_positions, shuffled_letters):
            out[idx] = amino_acid
        return "".join(out)
