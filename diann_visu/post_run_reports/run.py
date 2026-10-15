import argparse

from maplot.config import PlotProjectConfig
from maplot.project import PlotProject


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simple DIA-NN plotting scaffold")
    parser.add_argument(
        "input_file",
        nargs="?",
        default=None,
        help="Path to parquet/csv/tsv result file (optional when using experiment resolver)",
    )

    parser.add_argument(
        "--diann-out-root",
        default="./diann_out",
        help="Root folder that contains DIA-NN output versions",
    )
    parser.add_argument(
        "--pipeline-version",
        default="2_3_2",
        help="Pipeline version folder under --diann-out-root",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment folder name (e.g. maya_cele_46)",
    )
    parser.add_argument(
        "--quant-group",
        choices=["baseline", "tuned"],
        default="baseline",
        help="Choose baseline or tuned quant folder when resolving from experiment",
    )
    parser.add_argument(
        "--tuned-name",
        default=None,
        help="Exact tuned folder name, e.g. fast_rt_fr_lr_0p00011",
    )
    parser.add_argument(
        "--tuned-profile",
        default=None,
        help="Tuned profile prefix, e.g. fast_rt, balanced_rt, slow_rt",
    )
    parser.add_argument(
        "--fr",
        action="store_true",
        help="Use fr in tuned folder naming (<profile>_fr_lr_<lr>)",
    )
    parser.add_argument(
        "--lr",
        default=None,
        help="Learning rate token, supports both 0.00011 and 0p00011",
    )

    parser.add_argument("--outdir", default="./out", help="Output directory for plots")
    parser.add_argument("--qvalue", type=float, default=0.01, help="Q.Value filter threshold")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.input_file is None and args.experiment is None:
        parser.error("Provide either input_file or --experiment")

    config = PlotProjectConfig.from_input_or_experiment(
        output_dir=args.outdir,
        qvalue_threshold=args.qvalue,
        input_file=args.input_file,
        diann_out_root=args.diann_out_root,
        pipeline_version=args.pipeline_version,
        experiment=args.experiment,
        quant_group=args.quant_group,
        tuned_name=args.tuned_name,
        tuned_profile=args.tuned_profile,
        use_fr=args.fr,
        lr=args.lr,
    )

    project = PlotProject(config)
    project.register_default_plotters()
    outputs = project.run()

    print("Generated files:")
    for path in outputs:
        print(f"- {path}")


if __name__ == "__main__":
    main()
