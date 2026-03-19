from datetime import datetime
import multiprocessing as mp
import numpy as np

from circuit import predict, predict_batch, PARAM_NAMES, _pool_initializer
from data import make_synthetic_data, make_mnist_data
from optimiser import Adam, numerical_gradient, numerical_gradient_parallel
from utils import save_model, log_csv, MODELS_DIR
from profiling import timeit, timed

#calculate MSE = mean squared error between predictions and labels
@timeit("mse")
def mse(params, features, labels, pool=None):
    predictions = np.array(predict_batch(features, params, pool))
    return float(np.mean((predictions - labels) ** 2))

#calculate accuracy = ratio of correct predictions
def accuracy(params, features, labels, pool=None):
    predictions = np.array(predict_batch(features, params, pool))
    return float(np.mean(np.sign(predictions) == labels))

#train loop
def train(epochs=50, lr=0.05, seed=42, n_train=40, n_test=10,
          batch_size=0, log_every=10, use_mnist=False, save_path=None,
          n_workers=None):

    #random generators for data, initial params, and batch
    data_rng = np.random.default_rng(seed)
    param_rng = np.random.default_rng(seed + 1000)
    batch_rng = np.random.default_rng(seed + 2000)

    n_workers = n_workers or mp.cpu_count()

    #if manist, load and preprocess, else use synth data
    if use_mnist:
        X_tr, Y_tr, X_te, Y_te, preproc_state = make_mnist_data(n_train, n_test, data_rng)
    else:
        X_tr, Y_tr = make_synthetic_data(n_train, data_rng)
        X_te, Y_te = make_synthetic_data(n_test,  data_rng)
        preproc_state = {}

    #uniform random init of params in [-pi/2, pi/2]
    params = param_rng.uniform(-np.pi / 2, np.pi / 2, len(PARAM_NAMES))
    effective_bs = n_train if batch_size <= 0 else min(batch_size, n_train)
    optimizer = Adam(len(PARAM_NAMES), lr=lr)

    data_label = "binary mnist: 0 vs 1" if use_mnist else "synthetic"
    print(f"QCNN training:data={data_label}")
    print(f"seed={seed}  epochs={epochs}  lr={lr}  batch={effective_bs}")

    history = []
    
    #pool is created once and reused across all epochs — avoids per-epoch startup cost.
    with mp.Pool(processes=n_workers, initializer=_pool_initializer) as pool:
        for epoch in range(epochs):
            with timed("epoch [wall]"):

            #if there is a batch, sample it, otherwise use the whoel training set.
                if effective_bs < n_train:
                    batch_idx = batch_rng.choice(n_train, effective_bs, replace=False)
                    X_batch, Y_batch = X_tr[batch_idx], Y_tr[batch_idx]
                else:
                    X_batch, Y_batch = X_tr, Y_tr

            #compute grad: all 2*n_params*n_samples tasks dispatched in one pool.map call
                with timed("numerical gradient [epoch]"):
                    grad = numerical_gradient_parallel(params, X_batch, Y_batch, pool)
                params = optimizer.step(params, grad)

            #log progress every log_every epochs, and at the last epoch
                if epoch % log_every == 0 or epoch == epochs - 1:
                    with timed("mse and accuracy"):
                        train_loss = mse(params, X_tr, Y_tr, pool)
                        train_acc = accuracy(params, X_tr, Y_tr, pool)
                    print(f"epoch {epoch:3d}, loss {train_loss:.4f}, acc {train_acc:.3f}")

                    #append for later plot and save in model dict
                    history.append({"epoch": epoch, "loss": train_loss, "acc": train_acc})

        test_loss = mse(params, X_te, Y_te, pool)
        test_acc  = accuracy(params, X_te, Y_te, pool)
    print(f"\ntest loss: {test_loss:.4f}  |  test acc: {test_acc:.3f}")

    #save model and results
    model = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "data": "mnist" if use_mnist else "synthetic",
        "hyperparams": {"seed": seed, "epochs": epochs, "lr": lr, "n_train": n_train, "n_test": n_test, "batch": batch_size},
        "param_names": PARAM_NAMES,
        "params": params.tolist(),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": history,
        "preproc": preproc_state,
    }
    save_model(model, save_path)

    return {**model, "train": n_train, "test": n_test, "seed": seed, "epochs": epochs, "lr": lr}

#sweep over multiple hyperparameter configs, log results, and print ranked results by acc
def sweep(base_seed=42):

    configs = [{"epochs": 5, "lr": 0.05}, {"epochs": 5, "lr": 0.03},
               {"epochs": 5, "lr": 0.01}]
    
    results = []

    for i, cfg in enumerate(configs):
        run_seed = base_seed + i
        print(f"\n sweep run {i+1}/{len(configs)}  lr={cfg['lr']}  seed={run_seed}")
        result = train(epochs=cfg["epochs"], lr=cfg["lr"], seed=run_seed, save_path=MODELS_DIR / f"sweep_seed{run_seed}.json", log_every=cfg["epochs"])
        results.append(result)

    #sort by test acc (desc) and then test loss (asc)
    def sort_key(result):
        return (-result["test_acc"], result["test_loss"])
    
    results.sort(key=sort_key)
    
    print("\nsweep results (ranked by acc)\n")
    for rank, r in enumerate(results, 1):
        print(f"  {rank}. acc={r['test_acc']:.3f},  loss={r['test_loss']:.4f}, lr={r['lr']},  seed={r['seed']}")
    log_csv(results)
    return results
