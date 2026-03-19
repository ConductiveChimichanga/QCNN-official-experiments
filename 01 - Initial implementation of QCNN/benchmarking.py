import argparse
import time
from pathlib import Path

from profiling import print_report, reset_profiler, set_enabled, get_rows
from train import train as train_quokka
from qcnn_simple import train_simple


DEFAULTS = {
    "mnist": True,
    "train": 5,
    "test": 1,
    "epochs": 2,
    "lr": 0.05,
    "seed": 42,
    "batch": 0,
    "log_every": 10,
}


def _run_quokka(cfg):
    reset_profiler()
    set_enabled(True)
    t0 = time.perf_counter()
    result = train_quokka(
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        seed=cfg["seed"],
        n_train=cfg["train"],
        n_test=cfg["test"],
        batch_size=cfg["batch"],
        log_every=cfg["log_every"],
        use_mnist=cfg["mnist"],
        save_path=Path("models") / "benchmark_quokka.json",
    )
    wall = time.perf_counter() - t0
    rows = get_rows()
    print_report("\nquokka timing profile")
    return result, wall, rows


def _run_pennylane(cfg):
    reset_profiler()
    set_enabled(True)
    t0 = time.perf_counter()
    result = train_simple(
        epochs=cfg["epochs"],
        lr=cfg["lr"],
        seed=cfg["seed"],
        n_train=cfg["train"],
        n_test=cfg["test"],
        use_mnist=cfg["mnist"],
        log_every=cfg["log_every"],
    )
    wall = time.perf_counter() - t0
    rows = get_rows()
    print_report("\npennylane timing profile")
    return result, wall, rows


def _print_comparison(quokka_result, quokka_wall, pennylane_result, pennylane_wall):
    print("\nbackend comparison")
    print(f"{'backend':14s} {'wall_s':>10s} {'test_loss':>12s} {'test_acc':>10s}")
    print(
        f"{'quokka':14s} {quokka_wall:10.2f} "
        f"{quokka_result['test_loss']:12.4f} {quokka_result['test_acc']:10.3f}"
    )
    print(
        f"{'pennylane':14s} {pennylane_wall:10.2f} "
        f"{pennylane_result['test_loss']:12.4f} {pennylane_result['test_acc']:10.3f}"
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark quokka QCNN against qcnn_simple (PennyLane) with shared parameters"
    )
    parser.add_argument("--mnist", action="store_true", default=DEFAULTS["mnist"])
    parser.add_argument("--synthetic", action="store_true", help="override to synthetic data")
    parser.add_argument("--train", type=int, default=DEFAULTS["train"])
    parser.add_argument("--test", type=int, default=DEFAULTS["test"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    parser.add_argument("--log-every", type=int, default=DEFAULTS["log_every"], dest="log_every")
    return parser


def main():
    args = build_parser().parse_args()
    cfg = {
        "mnist": args.mnist and not args.synthetic,
        "train": args.train,
        "test": args.test,
        "epochs": args.epochs,
        "lr": args.lr,
        "seed": args.seed,
        "batch": args.batch,
        "log_every": args.log_every,
    }

    print("benchmark config")
    for key in ["mnist", "train", "test", "epochs", "lr", "seed", "batch", "log_every"]:
        print(f"  {key}: {cfg[key]}")

    quokka_result, quokka_wall, _ = _run_quokka(cfg)
    pennylane_result, pennylane_wall, _ = _run_pennylane(cfg)
    _print_comparison(quokka_result, quokka_wall, pennylane_result, pennylane_wall)


if __name__ == "__main__":
    main()