from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

#_emit is a helper that shows/saves a plot and prints the path
def _emit(fig, path, show):                        
    if show:
        plt.show()
    else:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  saved → {path}")
    plt.close(fig)

#plot no 1: training curves of loss and acc on y axis and epoch on x axis
def plot_training_curves(history, outdir, show=False):

    #extract epochs, losses, and accs from history
    epochs = []
    losses = []
    accs   = []
    for h in history:
        epochs.append(h["epoch"])
        losses.append(h["loss"])
        accs.append(h["acc"])


    #plot loss and acc curves side by side, with a shared x axis of epochs
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, losses, "o-", color="#2196F3", lw=2, ms=5)
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("mse loss")
    ax_loss.set_title("training loss"); ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(epochs, accs, "o-", color="#4CAF50", lw=2, ms=5)
    ax_acc.axhline(0.5, color="gray", ls="--", lw=1, label="random baseline")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("training accuracy"); ax_acc.grid(True, alpha=0.3); ax_acc.legend()

    fig.suptitle("QCNN — gradient descent progress", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _emit(fig, Path(outdir) / "training_curves.png", show)


#plot no 2: scatter of the 2d data colored by true label
def plot_scatter(X_2d, Y_te, Y_pred, outdir, show=False, data_src="synthetic"):
    correct = (Y_pred == Y_te)
    style_map = {
        (-1, True):  ("#1565C0", "o", "digit 0 T"),
        (-1, False): ("#90CAF9", "X", "digit 0 F"),
        ( 1, True):  ("#B71C1C", "o", "digit 1 T"),
        ( 1, False): ("#EF9A9A", "X", "digit 1 F"),
    }
    fig, ax = plt.subplots(figsize=(7, 6))
    for (label, ok), (color, marker, legend_label) in style_map.items():
        mask = (Y_te == label) & (correct == ok)
        if mask.any():
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, marker=marker,
                       s=85, label=legend_label, alpha=0.85, edgecolors="k", lw=0.4)

    ax.set_xlabel("pca dim 0" if data_src == "mnist" else "feature 0")
    ax.set_ylabel("pca dim 1" if data_src == "mnist" else "feature 1")
    ax.set_title("classification results — test set")
    ax.legend(loc="best", fontsize=9); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _emit(fig, Path(outdir) / "scatter.png", show)


#plot no 3: histogram of circuit scores, with different colors for the 2 classes
def plot_score_distribution(scores, Y_te, outdir, show=False):
    fig, ax = plt.subplots(figsize=(7, 4))
    for label, name, color in [(-1, "digit 0", "#1565C0"), (1, "digit 1", "#B71C1C")]:
        subset = scores[Y_te == label]
        if len(subset):
            ax.hist(subset, bins=min(15, len(subset)), alpha=0.65, color=color,
                    label=name, edgecolor="white", lw=0.5)
    ax.axvline(0, color="black", ls="--", lw=1.8, label="decision boundary")
    ax.set_xlabel("circuit expectation ⟨Z₀⟩"); ax.set_ylabel("count")
    ax.set_title("score distribution by class")
    ax.legend(); ax.grid(True, alpha=0.2)
    fig.tight_layout()
    _emit(fig, Path(outdir) / "score_distribution.png", show)



#plot no 4: bar plot of the final parameter values, colored by sign (positive blue, negative red)
def plot_param_values(params, param_names, outdir, show=False):
    x_pos     = np.arange(len(params))
    bar_color = []
    for v in params:
        bar_color.append("#2196F3" if v >= 0 else "#F44336")

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(x_pos, params, color=bar_color, edgecolor="k", lw=0.5)
    ax.set_xticks(x_pos); ax.set_xticklabels(param_names, rotation=30, ha="right", fontsize=8)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("parameter value (radians)")
    ax.set_title("trained circuit parameters")
    for bar, val in zip(bars, params):
        y_offset = 0.05 if val >= 0 else -0.12
        ax.text(bar.get_x() + bar.get_width() / 2, val + y_offset,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    _emit(fig, Path(outdir) / "params.png", show)
