import numpy as np

from circuit import predict, PARAM_NAMES
from data import make_synthetic_data
from utils import apply_preproc


#load mnist from openml and process correctly - pca, scaled [0, pi] and shuffled as in traioning
def _load_mnist_binary():
    try:
        from sklearn.datasets import fetch_openml
    except ImportError:
        raise RuntimeError("pip install scikit-learn  (needed for mnist probe)")
    print("loading MNIST")
    raw = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_full = raw.data
    y_full = raw.target.astype(int)
    mask = (y_full == 0) | (y_full == 1)
    return X_full[mask], y_full[mask]


def reconstruct_test_split(model, max_samples=None):

    #use the same seed and data for the test split
    hp = model.get("hyperparams", {})
    seed = hp.get("seed", 42)
    n_train = hp.get("n_train", 40)
    n_test = hp.get("n_test", 10)
    data_src = model.get("data", "synthetic")

    rng = np.random.default_rng(seed)

    if data_src == "mnist":
        X_raw, y_raw = _load_mnist_binary()
        shuffled_idx = rng.permutation(len(X_raw))
        X_raw, y_raw = X_raw[shuffled_idx], y_raw[shuffled_idx]
        X_features = apply_preproc(X_raw, model["preproc"])
        Y_labels = np.where(y_raw == 1, 1, -1).astype(float)

        pca_components = np.array(model["preproc"]["pca_components"])
        pca_mean = np.array(model["preproc"]["pca_mean"])
        X_2d = (X_raw[n_train:n_train + n_test] - pca_mean) @ pca_components.T  #for scatter

        X_te = X_features[n_train:n_train + n_test]
        Y_te = Y_labels  [n_train:n_train + n_test]
    else:
        #advance rng past training split
        make_synthetic_data(n_train, rng) 

        #get test split and return the 2 features for plotting           
        X_te, Y_te = make_synthetic_data(n_test, rng)
        X_2d = X_te                                  

    #if max_samples is set, cut down to max_samples
    if max_samples:
        cut = min(max_samples, len(X_te))
    else:
        cut = len(X_te)
    return X_te[:cut], Y_te[:cut], X_2d[:cut]


def classify_samples(params, X_te, Y_te):
    #run one circuit forward pass per sample and collect expectation values
    scores = []
    for i, sample in enumerate(X_te):
        score = predict(sample, params)
        scores.append(score)
        predicted_digit = "1" if score >= 0 else "0"
        print(f"[{i+1:>3}/{len(X_te)}], score={score:+.4f}, digit {predicted_digit}")

    scores = np.array(scores)
    Y_pred = np.sign(scores)

    #tie breaker edge case: if score is exactly 0, predict 1
    Y_pred[Y_pred == 0] = 1.0           
    is_correct = (Y_pred == Y_te)
    return scores, Y_pred, is_correct


def print_classification_report(Y_te, Y_pred, scores, is_correct):
    label = lambda v: "0" if v == -1 else "1"

    #print table of results per sample
    print(f"\n{'#':>3}  {'true':>5}  {'pred':>5}  {'score':>8}  result")
    print("─" * 38)
    for i, (true, pred, sc, ok) in enumerate(zip(Y_te, Y_pred, scores, is_correct)):
        print(f"{i:>3}  {label(true):>5}  {label(pred):>5}  {sc:>+8.4f}  {'T' if ok else 'F'}")

    #accuracy
    overall_acc = is_correct.mean()
    print(f"\naccuracy: {overall_acc:.3f}  ({is_correct.sum()}/{len(is_correct)} correct)")

    #confusion matrix
    tp = int(((Y_pred ==  1) & (Y_te ==  1)).sum())
    tn = int(((Y_pred == -1) & (Y_te == -1)).sum())
    fp = int(((Y_pred ==  1) & (Y_te == -1)).sum())
    fn = int(((Y_pred == -1) & (Y_te ==  1)).sum())

    print(f"\nconfusion matrix")
    print(f"             pred 0   pred 1")
    print(f"  true 0     {tn:>5}    {fp:>5}")
    print(f"  true 1     {fn:>5}    {tp:>5}")

    #precisoin and recal 
    safe = lambda a, b: a / b if b else float("nan")
    print(f"\ndigit 0  precision={safe(tn, tn+fn):.3f}  recall={safe(tn, tn+fp):.3f}")
    print(f"digit 1  precision={safe(tp, tp+fp):.3f}  recall={safe(tp, tp+fn):.3f}")

if __name__ == "__main__":
    Y_te = np.array([-1, -1, 1, 1])
    Y_pred = np.array([-1, 1, 1, -1])
    scores = np.array([-0.8, 0.2, 0.5, -0.3])
    is_correct = (Y_pred == Y_te)
    print_classification_report(Y_te, Y_pred, scores, is_correct)