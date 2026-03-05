# Experiment 01 — QCNN with Quokka + WMC

This folder is my first full QCNN implementation using Quokka. The model is small (4 qubits, 8 trainable parameters). 
For each forward pass, the circuit is converted to CNF and solved via weighted model counting to get exact probabilities, then we compute $\langle Z_0 \rangle = p(q_0=0)-p(q_0=1)$ as the score.

## What is in here

```
main.py          #entry point (train / probe)
circuit.py       #qasm render + quokka/wmc execution
train.py         #training loop and lr sweep
probe.py         #test replay + classification report
data.py          #synthetic + mnist(0/1) loader/preproc
optimiser.py     #adam + numerical gradient
plots.py         #png outputs for curves/scatter/hist/params
utils.py         #save/load model + csv logging
qasm/            #architecture + modular unitaries + assembler
models/          #saved checkpoints
results/         #run outputs

+ unittests      #details about testing below
```

## Setup

From repo root:

```bash
source code/venv/bin/activate
cd "experiments/official-experiments/01 - Initial implementation of QCNN"
```

If needed:

```bash
pip install scikit-learn
```

`config.json` points to GPMC using a relative path; `circuit.py` resolves it to absolute at runtime.

## Typical workflow

1) (optional) rebuild assembled qasm template

```bash
python qasm/assemble_qasm.py
```

2) train

```bash
#quick smoke run
python main.py train --epochs 2 --train 4 --test 2 --log-every 1 --save models/smoke_qcnn.json

#normal synthetic run
python main.py train --epochs 50 --lr 0.05 --train 40 --test 10

#mnist 0 vs 1
python main.py train --mnist --train 20 --test 10 --epochs 30
```

3) probe saved model

```bash
python main.py probe --model models/smoke_qcnn.json
python main.py probe --model models/smoke_qcnn.json --no-plots
```

Probe outputs go to `results/probe_output/` (or your custom `--outdir`):
- `training_curves.png`
- `scatter.png`
- `score_distribution.png`
- `params.png`

## Test commands

```bash
#fast/unit-level suite (includes assemble_qasm checks)
python -m unittest -v test_all_files_unittest.py

#slower integration test (train + probe via main.py)
python -m unittest -v test_integration_unittest.py

#run everything
python -m unittest -v test_all_files_unittest.py test_integration_unittest.py
```

## Technical walkthrough

- We encode each 4D input with `ry` gates on 4 qubits.
- Then we apply a QCNN-style stack: conv1 + pool1 + conv2 + pool2.
- The trainable angles are shared per layer.
- Each score evaluation does two WMC calls: one with $q_0=0$, one with $q_0=1$.
- Training uses central-difference numerical gradients and Adam.
- Probing recreates the exact test split from the saved seed/preprocessing state.

## Notes 

- WMC is exact but can be slow; keep train/test sizes small while iterating.
- MNIST accuracy varies here by design (4 PCA components, tiny QCNN).

