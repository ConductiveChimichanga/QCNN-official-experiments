import subprocess
import sys
import unittest
from pathlib import Path


#slow integration tests that run the whole workflow on small epoch and samples
HERE = Path(__file__).resolve().parent


class TestIntegrationWorkflow(unittest.TestCase):

    #run train then probe from main.py with tiny settings
    def test_main_train_then_probe(self):
        model_path = HERE / "models" / "smoke_integration.json"
        outdir = HERE / "results" / "smoke_integration_probe"

        train_cmd = [
            sys.executable,
            "main.py",
            "train",
            "--epochs", "1",
            "--train", "2",
            "--test", "1",
            "--log-every", "1",
            "--save", str(model_path),
            "--csv", str(HERE / "results" / "smoke_integration_train.csv"),
        ]
        train_proc = subprocess.run(
            train_cmd,
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=240,
        )
        self.assertEqual(train_proc.returncode, 0, msg=train_proc.stdout + "\n" + train_proc.stderr)
        self.assertTrue(model_path.exists())

        probe_cmd = [
            sys.executable,
            "main.py",
            "probe",
            "--model", str(model_path),
            "--no-plots",
            "--n", "1",
            "--outdir", str(outdir),
        ]
        probe_proc = subprocess.run(
            probe_cmd,
            cwd=HERE,
            capture_output=True,
            text=True,
            timeout=240,
        )
        self.assertEqual(probe_proc.returncode, 0, msg=probe_proc.stdout + "\n" + probe_proc.stderr)
        self.assertIn("QCNN probe", probe_proc.stdout)


if __name__ == "__main__":
    unittest.main()
