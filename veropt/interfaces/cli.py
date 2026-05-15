"""
Command-line interface for veropt experiment management utilities.

Usage
-----
Roll back an experiment to a specific point:

    python -m veropt.interfaces.cli rollback \\
        --config path/to/experiment_config.json \\
        --target-point 4 \\
        [--simulation-folders rename|delete|force_delete]

Arguments
---------
rollback
    --config              Path to the experiment config JSON.
    --target-point        Point index to roll back to (must be on a batch boundary).
    --simulation-folders  How to handle simulation directories for rolled-back points.
                          Options: rename (default), delete (with prompt), force_delete.
"""

import argparse
import sys


def _make_rollback_parser(subparsers: argparse.Action) -> None:
    rollback_parser = subparsers.add_parser(  # type: ignore[attr-defined]
        "rollback",
        help="Roll back an experiment to a specific point.",
        description="Roll back an experiment's state, optimiser, and simulation directories to a given point.",
    )
    rollback_parser.add_argument(
        "--config",
        required=True,
        help="Path to the experiment config JSON.",
    )
    rollback_parser.add_argument(
        "--target-point",
        required=True,
        type=int,
        help="Point index to roll back to (must be a multiple of n_evaluations_per_step).",
    )
    rollback_parser.add_argument(
        "--simulation-folders",
        choices=["rename", "delete", "force_delete"],
        default="rename",
        help="What to do with simulation directories for rolled-back points. "
             "rename = rename with _rolled_back suffix (default). "
             "delete = ask for confirmation then delete. "
             "force_delete = delete without asking.",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m veropt.interfaces.cli",
        description="veropt experiment management utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _make_rollback_parser(subparsers)

    args = parser.parse_args()

    if args.command == "rollback":
        from veropt.interfaces.rollback import rollback_experiment
        rollback_experiment(
            experiment_config_path=args.config,
            target_point=args.target_point,
            simulation_folder_handling=args.simulation_folders,
        )
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
