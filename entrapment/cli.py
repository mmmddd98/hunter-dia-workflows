from argparse import ArgumentParser
from pathlib import Path

from .core import EntrapmentGenerator


def main() -> None:
    parser = ArgumentParser(description="Create combined original+entrapment FASTA files.")
    parser.add_argument("--fasta", required=True, help="Original input FASTA path.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--filename", required=True, help="Output FASTA filename.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--keep-first-letter", action="store_true", default=True)
    parser.add_argument("--shuffle-first-letter", dest="keep_first_letter", action="store_false")
    parser.add_argument(
        "--shuffle-mode",
        choices=("fixed-r", "simple"),
        default="fixed-r",
        help="fixed-r keeps arginine positions fixed; simple is the legacy whole-tail shuffle.",
    )
    parser.add_argument(
        "--legacy-simple-shuffle",
        dest="shuffle_mode",
        action="store_const",
        const="simple",
        help="Use the previous shuffle behaviour.",
    )
    parser.add_argument(
        "--shuffled-only",
        action="store_true",
        help="Write only shuffled proteins instead of original+shuffled.",
    )

    args = parser.parse_args()
    shuffled, out_path = EntrapmentGenerator().entrap(
        fasta_path=Path(args.fasta),
        out_dir=Path(args.out_dir),
        filename=args.filename,
        seed=args.seed,
        keep_first_letter=args.keep_first_letter,
        shuffle_mode=args.shuffle_mode,
        combine_with_original=not args.shuffled_only,
    )

    print(f"{out_path}\tshuffled={len(shuffled)}\tmode={args.shuffle_mode}")


if __name__ == "__main__":
    main()
