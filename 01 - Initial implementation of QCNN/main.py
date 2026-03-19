#external libraries

import sys, argparse
from pathlib import Path
import matplotlib
import numpy as np

#internal libraries

from train import train, sweep
from utils import log_csv
from utils  import load_model, RESULTS_DIR
from probe  import reconstruct_test_split, classify_samples, print_classification_report
from plots  import (plot_training_curves, plot_scatter,
                        plot_score_distribution, plot_param_values)
from profiling import set_enabled, reset_profiler, print_report


HERE = Path(__file__).parent


#comand-line training
def cmd_train(args):
    if args.timeit:
        reset_profiler()
    set_enabled(args.timeit)
    
    #if sweep is set, other flags are ignored
    if args.sweep:
        sweep(args.seed)
        return

    #arguments: seed, epochs, lr, n_train, n_test, batch_size, log_every, use_mnist, save_path, csv_path
    result = train(
        epochs=args.epochs, lr=args.lr, seed=args.seed,
        n_train=args.train, n_test=args.test,
        batch_size=args.batch, log_every=args.log_every,
        use_mnist=args.mnist, save_path=args.save,
    )
    log_csv([result], args.csv)
    if args.timeit:
        print_report("\ntraining timing profile")


#comand-line probe and plot. probe runs the circuit on test again and compares to saved results, then generates plots
def cmd_probe(args):

    #tkagg makes sure it doesnt open plots
    matplotlib.use("TkAgg" if args.show else "Agg")

    #load model and reconstruct the test using the same params
    model=load_model(args.model)
    params=np.array(model["params"])
    param_names=model.get("param_names", [])
    data_src=model.get("data", "synthetic")
    history=model.get("history", [])

    print(f"QCNN probe - data={data_src}, model={args.model}")

    #print hyperparams and saved results
    hp = model.get("hyperparams", {})
    if hp:
        print(f"seed={hp.get('seed')}  epochs={hp.get('epochs')}  lr={hp.get('lr')}"
              f"  n_train={hp.get('n_train')}  n_test={hp.get('n_test')}")
    if model.get("test_acc") is not None:
        print(f"saved result  acc={model['test_acc']:.3f}  loss={model['test_loss']:.4f}")
    print("=" * 60 + "\n")

    X_te, Y_te, X_2d = reconstruct_test_split(model, args.n)
    print(f"test samples to probe: {len(X_te)}\n")

    scores, Y_pred, is_correct = classify_samples(params, X_te, Y_te)
    print_classification_report(Y_te, Y_pred, scores, is_correct)

    if args.no_plots:
        return

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = RESULTS_DIR / "probe_output"
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nsaving plots to {outdir}/")

    plot_param_values(params, param_names, outdir, args.show)
    plot_score_distribution(scores, Y_te, outdir, args.show)
    if history:
        plot_training_curves(history, outdir, args.show)
    if X_2d.shape[1] >= 2:
        plot_scatter(X_2d, Y_te, Y_pred, outdir, args.show, data_src)
    print("done probing and plotting!")

#parser for command-line arguments, with two subcommands: train and probe
def build_parser():

    parser = argparse.ArgumentParser(
        prog="main.py",
        description="qcnn — train or probe a qcnn via quokka"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    #training
    train_parser = subparsers.add_parser(
        "train",
        help="train the qcnn and save a model checkpoint"
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--lr", type=float, default=0.05)
    train_parser.add_argument("--train", type=int, default=40)
    train_parser.add_argument("--test", type=int, default=10)
    train_parser.add_argument("--batch", type=int, default=0, help="0 = full-batch")
    train_parser.add_argument("--log-every", type=int, default=10, dest="log_every")
    train_parser.add_argument("--mnist", action="store_true", help="use mnist 0 vs 1")
    train_parser.add_argument("--sweep", action="store_true", help="run lr sweep")
    train_parser.add_argument("--save", type=str, default=None)
    train_parser.add_argument("--csv", type=str, default=None)
    train_parser.add_argument("--timeit", action="store_true", help="collect and print timing table")

    #probe and plot
    probe_parser = subparsers.add_parser(
        "probe",
        help="classify test samples and generate plots"
    )
    probe_parser.add_argument(
        "--model",
        type=str,
        default=str(HERE / "models" / "qcnn_trained.json")
    )
    probe_parser.add_argument("--n", type=int, default=None, help="limit to first n samples")
    probe_parser.add_argument("--outdir", type=str, default=None)
    probe_parser.add_argument("--no-plots", action="store_true", dest="no_plots")
    probe_parser.add_argument("--show", action="store_true", help="interactive plot windows")

    return parser

#main, executes the command based on args
#eg: `python main.py train --epochs 100 --lr 0.01` or `python main.py probe --model models/qcnn_trained.json`
if __name__ == "__main__":
    args = build_parser().parse_args()
    {"train": cmd_train, "probe": cmd_probe}[args.command](args)
