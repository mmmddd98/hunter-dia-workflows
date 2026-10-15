# Entrapment FASTA generation

Generate shuffled protein sequences for entrapment analyses. By default, the
output combines the original sequences and shuffled entries prefixed `ENT_`.
Arginine positions and the first residue are preserved; the other residues
are shuffled with a reproducible seed.

## Input

The parser requires UniProt-style headers with `OS`, `OX`, `PE` and `SV` fields
(`GN` is optional), for example:

```text
>sp|P00001|EXAMPLE_HUMAN Example protein OS=Homo sapiens OX=9606 GN=EXAMPLE PE=1 SV=1
MACDEFGHRKLMNPQRSTDAVK
```

Other header formats are skipped with a `Could not parse header` message.
Check that the reported shuffled count matches the number of intended input
proteins. Generic FASTA headers are not supported by this parser.

## Run

Use Python 3.10 or newer. No third-party packages are required. From the
repository root, replace the input path with your own UniProt FASTA:

```bash
python -m entrapment.cli \
  --fasta inputs/proteins.fasta \
  --out-dir outputs/entrapment \
  --filename proteins_with_entrapment.fasta \
  --seed 1337
```

`--shuffled-only` omits the original entries. `--shuffle-mode simple` shuffles
without fixing arginine positions; `--shuffle-first-letter` also releases the
first residue. Run `python -m entrapment.cli --help` for all options.

Keep input databases and generated FASTA files outside version control.
