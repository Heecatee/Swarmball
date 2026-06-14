import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create training progress plots from Stable-Baselines3 monitor CSV files."
    )

    parser.add_argument(
        "--runs-dir",
        type=str,
        default="runs",
        help="Directory containing training run folders.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Directory where plots should be saved.",
    )

    parser.add_argument(
        "--window",
        type=int,
        default=10,
        help="Rolling window size for smoothing rewards.",
    )

    parser.add_argument(
        "--include-eval",
        action="store_true",
        help="Also plot eval_monitor files. By default only training monitor files are used.",
    )

    return parser.parse_args()


def find_monitor_files_by_run(runs_dir: Path, include_eval: bool):
    """
    Finds monitor CSV files recursively and groups them by run directory.

    Supported examples:
    runs/run_name/monitor/env_0.monitor.csv
    runs/run_name/monitor/env_1.monitor.csv
    runs/run_name/eval_monitor/env_0.monitor.csv
    runs/run_name/training.monitor.csv
    runs/run_name/monitor.monitor.csv
    """
    files_by_run = defaultdict(list)

    for file_path in runs_dir.rglob("*.monitor.csv"):
        relative_parts = file_path.relative_to(runs_dir).parts

        if len(relative_parts) < 2:
            continue

        run_name = relative_parts[0]

        # By default, skip eval_monitor because we want the training curve.
        if not include_eval and "eval_monitor" in relative_parts:
            continue

        files_by_run[run_name].append(file_path)

    return dict(files_by_run)


def read_monitor_csv(file_path: Path):
    """
    Reads Stable-Baselines3 Monitor CSV.

    Monitor files usually start with metadata line:
    #{"t_start": ...}

    Then they contain columns:
    r,l,t

    r - episode reward
    l - episode length
    t - elapsed time
    """
    return pd.read_csv(file_path, comment="#")


def prepare_training_curve(monitor_files: list[Path]):
    frames = []

    for file_path in monitor_files:
        df = read_monitor_csv(file_path)

        required_columns = {"r", "l"}
        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Monitor CSV must contain columns {required_columns}, "
                f"but got {set(df.columns)} in {file_path}"
            )

        df = df.copy()
        df["source_file"] = str(file_path)
        frames.append(df)

    if not frames:
        raise ValueError("No monitor files provided.")

    df = pd.concat(frames, ignore_index=True)

    # If Monitor has elapsed time column, use it to sort episodes from parallel envs.
    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)

    # Approximate global training timesteps as cumulative episode lengths.
    df["timesteps"] = df["l"].cumsum()

    return df


def plot_single_run(run_name: str, monitor_files: list[Path], output_dir: Path, window: int):
    df = prepare_training_curve(monitor_files)

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["timesteps"],
        df["r"],
        alpha=0.3,
        label="Raw episode reward",
    )

    if len(df) >= window:
        rolling_reward = df["r"].rolling(window=window).mean()

        plt.plot(
            df["timesteps"],
            rolling_reward,
            linewidth=2,
            label=f"Rolling mean ({window})",
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Episode reward")
    plt.title(f"Training progress: {run_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / f"{run_name}_training_progress.png"
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved plot: {output_path}")


def main():
    args = parse_args()

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        print(f"Runs directory does not exist: {runs_dir}")
        sys.exit(1)

    files_by_run = find_monitor_files_by_run(
        runs_dir=runs_dir,
        include_eval=args.include_eval,
    )

    if not files_by_run:
        print(f"No monitor CSV files found in: {runs_dir}")
        print("Expected files like:")
        print("  runs/<run_name>/monitor/env_0.monitor.csv")
        print("  runs/<run_name>/monitor/env_1.monitor.csv")
        print("  runs/<run_name>/training.monitor.csv")
        sys.exit(1)

    print("Found monitor files:")
    for run_name, files in files_by_run.items():
        print(f"- {run_name}:")
        for file_path in files:
            print(f"  - {file_path}")

    for run_name, monitor_files in files_by_run.items():
        plot_single_run(
            run_name=run_name,
            monitor_files=monitor_files,
            output_dir=output_dir,
            window=args.window,
        )


if __name__ == "__main__":
    main()