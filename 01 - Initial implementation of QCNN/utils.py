import json, csv
from pathlib import Path
from datetime import datetime
import numpy as np

HERE = Path(__file__).parent
MODELS_DIR  = HERE / "models"
RESULTS_DIR = HERE / "results"

_LEGACY_PARAM_NAMES = ["c1_rx0", "c1_rx1", "p1_crz_angle", "p1_crx_angle",
                       "c2_rx0", "c2_rx1", "p2_crz_angle", "p2_crx_angle"]

_CSV_FIELDS = ["timestamp", "seed", "epochs", "lr", "train", "test",
               "test_loss", "test_acc", "data"]


#load model from json
#supports both new rich format and old dicts, from a previous ver
def load_model(path):
    raw = json.loads(Path(path).read_text())
    if "params" in raw:
        return raw
    legacy_params = []  #old flat format
    for k in _LEGACY_PARAM_NAMES:
        legacy_params.append(raw[k])
    return {"params": legacy_params, "param_names": _LEGACY_PARAM_NAMES, "data": "synthetic", "history": [], "preproc": {}, "hyperparams": {"seed": 42, "n_train": 40, "n_test": 10}, "test_loss": None, "test_acc": None}


#save model dict to json
def save_model(model_dict, path=None):
    out = Path(path) if path else MODELS_DIR / "qcnn_trained.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(model_dict, indent=2))
    print(f"model saved to {out}")
    return out


#appen results to csv file
def log_csv(results, path=None):

    #append one row per result dict; creates file + header if missing
    dest = Path(path) if path else RESULTS_DIR / "results.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    file_exists = dest.exists()
    with open(dest, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for r in results:
            row = {}
            for k in _CSV_FIELDS:
                row[k] = r.get(k, "")
            row["timestamp"] = datetime.now().isoformat(timespec="seconds")
            writer.writerow(row)

    print(f"results logged to {dest}")


#reuse the same preproc steps as in training
def apply_preproc(X_raw, preproc):
    components = np.array(preproc["pca_components"])    
    mean_vec = np.array(preproc["pca_mean"])         
    data_min = np.array(preproc["scaler_data_min"])
    data_max = np.array(preproc["scaler_data_max"])
    lo, hi = preproc.get("feature_range", [0, float(np.pi)])

    X_projected = (X_raw - mean_vec) @ components.T  #project to 2d
    X_normalised = (X_projected - data_min) / (data_max - data_min)
    return X_normalised * (hi - lo) + lo #scale to [0, pi]
