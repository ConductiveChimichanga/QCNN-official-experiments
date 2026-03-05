import numpy as np
from pathlib import Path

#load mnist and pca, if not vailable, make synthetic data
try:
    from sklearn.datasets import fetch_openml
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import MinMaxScaler
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


#synth data: 4 features in [0, pi], label is 1 if sum > 2pi else -1
def make_synthetic_data(n_samples, rng):
    features = rng.uniform(0, np.pi, (n_samples, 4))
    labels = np.where(np.sum(features, axis=1) > 2 * np.pi, 1, -1)
    return features, labels


#mnist data
#get mnist 0 vs 1, compress to n_components via pca, scale to [0, pi]
def make_mnist_data(n_train, n_test, rng, n_components=4):

    #install warning
    if not _SKLEARN:
        raise RuntimeError("pip install scikit-learn  (needed for mnist)")

    print("loading MNIST")
    raw = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
    X_full, y_full = raw.data, raw.target.astype(int)


    #get only 0 and 1
    binary_mask = (y_full == 0) | (y_full == 1)
    X_bin, y_bin = X_full[binary_mask], y_full[binary_mask]

    #pca to 4 components for 4 qubits
    pca = PCA(n_components=n_components, random_state=0)
    X_pca = pca.fit_transform(X_bin)

    #scale to [0, pi]
    scaler = MinMaxScaler(feature_range=(0, np.pi))
    X_scaled = scaler.fit_transform(X_pca).astype(float)
    labels = np.where(y_bin == 1, 1, -1).astype(float)

    #shuffle
    shuffled_idx = rng.permutation(len(X_scaled))
    X_scaled, labels = X_scaled[shuffled_idx], labels[shuffled_idx]
    
    #check nough samples for the requestd train/test
    total_needed = n_train + n_test
    if total_needed > len(X_scaled):
        raise ValueError(f"asked for {total_needed} samples, only {len(X_scaled)} available")

    ev_ratio = pca.explained_variance_ratio_.sum()
    print(f"mnist ready: {n_train} train + {n_test} test")
    print(f"(pca {X_bin.shape[1]} to {n_components})")
    #print(f"variance ratio: {ev_ratio:.4f} ({ev_ratio * 100:.2f}%)")
    
    #prepare preprocessing state to be saved for reproducibility with probe
    preproc_state = {
        "type": "pca_minmax",
        "n_components": n_components,
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "scaler_data_min": scaler.data_min_.tolist(),
        "scaler_data_max": scaler.data_max_.tolist(),
        "feature_range": [0, float(np.pi)],
    }

    #return train/test split
    X_tr = X_scaled[:n_train];       Y_tr = labels[:n_train]
    X_te = X_scaled[n_train:total_needed]; Y_te = labels[n_train:total_needed]
    return X_tr, Y_tr, X_te, Y_te, preproc_state
