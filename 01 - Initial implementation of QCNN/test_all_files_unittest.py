import unittest
import tempfile
import importlib.util
from pathlib import Path

import numpy as np

import data
import optimiser
import utils
import plots
import probe
import circuit
import train


#root of experiment 01
HERE = Path(__file__).resolve().parent


class TestAllPythonFiles(unittest.TestCase):

    #make sure every python file in this experiment at least parses
    def test_all_python_files_compile(self):
        py_files = sorted(HERE.rglob("*.py"))
        self.assertGreater(len(py_files), 0)

        for file_path in py_files:
            source = file_path.read_text(encoding="utf-8")
            try:
                compile(source, str(file_path), "exec")
            except SyntaxError as exc:
                self.fail(f"syntax error in {file_path}: {exc}")

    #explicit check requested: assemble_qasm must build an output file
    def test_assemble_qasm_builds_template(self):
        asm_path = HERE / "qasm" / "assemble_qasm.py"
        spec = importlib.util.spec_from_file_location("assemble_qasm", asm_path)
        asm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(asm)

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "unitaries_test.qasm"
            asm.build_qasm_template(
                str(HERE / "qasm" / "unitaries"),
                str(HERE / "qasm" / "architecture_simple.json"),
                str(out),
            )
            self.assertTrue(out.exists())
            text = out.read_text(encoding="utf-8")
            self.assertIn("OPENQASM 2.0;", text)
            self.assertIn("qreg q[4];", text)
            self.assertIn("{{c1_", text)

    #circuit render path should include encoding and qasm header
    def test_circuit_builds_qasm_string(self):
        x = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
        params = np.zeros(len(circuit.PARAM_NAMES), dtype=float)
        qasm = circuit.build_circuit(x, params)

        self.assertIn("OPENQASM 2.0;", qasm)
        self.assertIn("qreg q[4];", qasm)
        self.assertIn("ry(0.1) q[0];", qasm)

    #probe classify should apply tie-break rule score==0 => class 1
    def test_probe_classify_tie_break(self):
        original_predict = probe.predict
        try:
            scores = iter([0.0, -0.2, 0.4])
            probe.predict = lambda _x, _p: next(scores)

            params = np.zeros(8, dtype=float)
            X_te = np.zeros((3, 4), dtype=float)
            Y_te = np.array([1.0, -1.0, 1.0], dtype=float)

            raw_scores, y_pred, is_ok = probe.classify_samples(params, X_te, Y_te)
            np.testing.assert_allclose(raw_scores, np.array([0.0, -0.2, 0.4]))
            np.testing.assert_allclose(y_pred, np.array([1.0, -1.0, 1.0]))
            self.assertTrue(bool(is_ok.all()))
        finally:
            probe.predict = original_predict

    #train helper metrics should behave on a mocked predictor
    def test_train_mse_and_accuracy(self):
        original_predict = train.predict
        try:
            train.predict = lambda x, _p: 1.0 if x[0] > 0 else -1.0
            X = np.array([[1, 0, 0, 0], [-1, 0, 0, 0]], dtype=float)
            Y = np.array([1.0, -1.0], dtype=float)

            params = np.zeros(8, dtype=float)
            self.assertAlmostEqual(train.mse(params, X, Y), 0.0, places=8)
            self.assertAlmostEqual(train.accuracy(params, X, Y), 1.0, places=8)
        finally:
            train.predict = original_predict

    #utils save/load/log should work with temp paths
    def test_utils_roundtrip_and_csv_log(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            model_path = td_path / "model.json"
            csv_path = td_path / "results.csv"

            model = {
                "params": [0.1] * 8,
                "param_names": [f"p{i}" for i in range(8)],
                "data": "synthetic",
                "history": [],
                "preproc": {},
                "hyperparams": {"seed": 1, "n_train": 2, "n_test": 1},
                "test_loss": 0.5,
                "test_acc": 1.0,
            }

            utils.save_model(model, model_path)
            loaded = utils.load_model(model_path)
            self.assertEqual(len(loaded["params"]), 8)

            utils.log_csv([
                {
                    "seed": 1,
                    "epochs": 2,
                    "lr": 0.05,
                    "train": 2,
                    "test": 1,
                    "test_loss": 0.5,
                    "test_acc": 1.0,
                    "data": "synthetic",
                }
            ], path=csv_path)
            self.assertTrue(csv_path.exists())
            self.assertIn("test_acc", csv_path.read_text(encoding="utf-8"))

    #plots should write all expected files for tiny fake inputs
    def test_plots_write_files(self):
        with tempfile.TemporaryDirectory() as td:
            outdir = Path(td)
            history = [{"epoch": 0, "loss": 1.0, "acc": 0.5}, {"epoch": 1, "loss": 0.8, "acc": 0.75}]
            scores = np.array([-0.3, 0.4], dtype=float)
            y_true = np.array([-1.0, 1.0], dtype=float)
            y_pred = np.array([-1.0, 1.0], dtype=float)
            X2 = np.array([[0.0, 0.1], [0.2, 0.3]], dtype=float)
            params = np.array([0.1] * 8, dtype=float)
            names = [f"p{i}" for i in range(8)]

            plots.plot_training_curves(history, outdir, show=False)
            plots.plot_score_distribution(scores, y_true, outdir, show=False)
            plots.plot_scatter(X2, y_true, y_pred, outdir, show=False)
            plots.plot_param_values(params, names, outdir, show=False)

            self.assertTrue((outdir / "training_curves.png").exists())
            self.assertTrue((outdir / "score_distribution.png").exists())
            self.assertTrue((outdir / "scatter.png").exists())
            self.assertTrue((outdir / "params.png").exists())

    #qcnn_demo should generate a qasm output file quickly
    def test_qcnn_demo_generate_circuit(self):
        import qcnn_demo

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "demo.qasm"
            x = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
            params = {
                "c1_rx0": 0.1, "c1_rx1": 0.2,
                "p1_crz_angle": 0.3, "p1_crx_angle": 0.4,
                "c2_rx0": 0.5, "c2_rx1": 0.6,
                "p2_crz_angle": 0.7, "p2_crx_angle": 0.8,
            }
            text = qcnn_demo.generate_circuit(x, params, out)
            self.assertTrue(out.exists())
            self.assertIn("OPENQASM 2.0;", text)

    #optimiser module should expose expected primitives
    def test_optimiser_symbols(self):
        self.assertTrue(callable(optimiser.numerical_gradient))
        self.assertTrue(hasattr(optimiser, "Adam"))

    #numerical grad should match analytic grad on an easy quadratic
    def test_numerical_gradient_quadratic(self):
        def loss_fn(p):
            return float(np.sum((p - 2.0) ** 2))

        params = np.array([0.5, -1.0, 3.0], dtype=float)
        grad_num = optimiser.numerical_gradient(loss_fn, params, eps=1e-5)
        grad_true = 2.0 * (params - 2.0)
        np.testing.assert_allclose(grad_num, grad_true, atol=1e-4, rtol=1e-4)

    #adam first update should move params opposite to positive gradient
    def test_adam_step_direction(self):
        optimizer = optimiser.Adam(n_params=3, lr=0.1)
        params = np.array([0.0, 0.0, 0.0], dtype=float)
        grad = np.array([1.0, 1.0, 1.0], dtype=float)

        updated = optimizer.step(params, grad)
        self.assertTrue(np.all(updated < params))
        np.testing.assert_allclose(updated, np.array([-0.1, -0.1, -0.1]), atol=1e-6)

    #data module should return expected synthetic format
    def test_data_synthetic_quick(self):
        rng = np.random.default_rng(10)
        X, Y = data.make_synthetic_data(5, rng)
        self.assertEqual(X.shape, (5, 4))
        self.assertEqual(Y.shape, (5,))

    #quick sanity check for synthetic data shape/range/labels
    def test_synthetic_data_basic_properties(self):
        rng = np.random.default_rng(7)
        features, labels = data.make_synthetic_data(12, rng)

        self.assertEqual(features.shape, (12, 4))
        self.assertEqual(labels.shape, (12,))
        self.assertTrue(np.all(features >= 0.0))
        self.assertTrue(np.all(features <= np.pi))

        unique_labels = set(np.unique(labels).tolist())
        self.assertTrue(unique_labels.issubset({-1, 1}))


if __name__ == "__main__":
    unittest.main()
