import json
import os
import re
import ast
import tempfile
import importlib.util
import operator as op
from pathlib import Path
import numpy as np


HERE = Path(__file__).parent
QASM_DIR = HERE / "qasm"

PARAM_NAMES = ["c1_rx0", "c1_rx1", "p1_crz_angle", "p1_crx_angle",
               "c2_rx0", "c2_rx1", "p2_crz_angle", "p2_crx_angle"]


#load config.json from the repo root and point quokka at it
#if the file is missing, quokka falls back to its defaults (gpmc must be in PATH)
def _patch_quokka_config():
    cfg_path = HERE / "config.json"
    if not cfg_path.exists():
        return

    cfg = json.loads(cfg_path.read_text())
    cfg["DEBUG"] = False

    #resolve a relative gpmc path to an absolute one
    parts = cfg.get("ToolInvocation", "").split(" ", 1)
    if parts[0] and not os.path.isabs(parts[0]):
        parts[0] = str((HERE / parts[0]).resolve())
        cfg["ToolInvocation"] = " ".join(parts)

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, tmp)
    tmp.close()
    os.environ["QUOKKA_CONFIG"] = tmp.name


#run the patch at import so all calls use quokka correctly
_patch_quokka_config()

#import quokka after patching the config
import quokka_sharp as qk
import quokka_sharp.sim as _sim


#snapshot the tool command now, before the upstream library can mutate it
#this avoids a bug where quokka_sharp.sim converts the command string to a list in-place
_TOOL_CMD = str(_sim.CONFIG["ToolInvocation"])


def _stable_wmc(path, square):
    proc = _sim.Popen(_TOOL_CMD.split() + [path], stdout=_sim.PIPE)
    try:
        return _sim.parse_wmc_result(proc.communicate(timeout=_sim.TIMEOUT), square)
    except _sim.TimeoutExpired:
        os.kill(proc.pid, 9)
        return "TIMEOUT"

_sim.WMC = _stable_wmc


#qasm template — compiled once at import time from the unitaries folder
def _compile_qasm_template():
    spec = importlib.util.spec_from_file_location("asm", QASM_DIR / "assemble_qasm.py")
    asm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(asm)

    tpl = QASM_DIR / "unitaries_simple.qasm"
    asm.build_qasm_template(
        str(QASM_DIR / "unitaries"),
        str(QASM_DIR / "architecture_simple.json"),
        str(tpl)
    )
    return tpl.read_text()

_TEMPLATE = _compile_qasm_template()

#arithmetic operator map used when evaluating angle expressions
_ANGLE_OPS = {
    ast.Add:  op.add,
    ast.Sub:  op.sub,
    ast.Mult: op.mul,
    ast.Div:  op.truediv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


#function that evaluates strings that may contain pi or basic arithmetic 
def _eval_gate_angle(expr):
    #eg "pi/2 + 0.1"
    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name) and node.id == "pi":
            return float(np.pi)
        if isinstance(node, ast.BinOp):
            return _ANGLE_OPS[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp):
            return _ANGLE_OPS[type(node.op)](walk(node.operand))
        raise ValueError(expr)
    return walk(ast.parse(expr, mode="eval"))

def _render_qasm(template, param_dict):

    #first transform: fill {{param_name}} placeholders with their numeric values
    #eg "rx({{c1_rx0/2}})" becomes "rx(0.15/2)"
    filled = re.sub(r"\{\{(\w+)\}\}", lambda m: str(param_dict[m.group(1)]), template)

    #resolve any expressions (e.g. pi/2) to plain floats
    #eg "rx(0.15/2)" becomes "rx(0.075)"
    def fix_angle(m):
        gate = m.group(1)
        expr = m.group(2).strip()
        qubit = m.group(3)
        if any(c in expr for c in "+-*/") or "pi" in expr:
            expr = format(_eval_gate_angle(expr), ".15g")
        return f"{gate}({expr}) {qubit};"

    return re.sub(r"\b([a-z]+)\(([^)]+)\)\s+(q\[\d+\]);", fix_angle, filled)


def build_circuit(features, params):
    #render the parameterised template, then insert ry encoding gates after the qreg declaration
    rendered = _render_qasm(_TEMPLATE, dict(zip(PARAM_NAMES, params)))
    lines = []
    for l in rendered.split("\n"):
        if not l.strip().startswith("//"):
            lines.append(l)

    #find the header end (line with qreg declaration)
    header_end = None
    for i, line in enumerate(lines):
        if "qreg q[" in line:
            header_end = i + 1
            break
    
    #build the encoding gates
    encoders = []
    for i, v in enumerate(features):
        encoders.append(f"ry({v}) q[{i}];")

    return "\n".join(lines[:header_end] + encoders + lines[header_end:])


def run_quokka(qasm_str):
    #evaluate the expectation value <Z_0> = p(q0=0) - p(q0=1) using weighted model counting
    with tempfile.TemporaryDirectory(prefix="qcnn_") as td:
        qf = Path(td) / "circ.qasm"
        qf.write_text(qasm_str)

        circuit = qk.encoding.QASMparser(str(qf), translate_ccx=True)

        #evaluating q0 is 0
        cnf0 = qk.encoding.QASM2CNF(circuit, computational_basis=True)
        cnf0.add_measurement({0: 0.0})
        p0 = qk.Simulate(cnf0, cnf_file_root=td)

        #evaluating q0 is 1
        cnf1 = qk.encoding.QASM2CNF(circuit, computational_basis=True)
        cnf1.add_measurement({0: 1.0})
        p1 = qk.Simulate(cnf1, cnf_file_root=td)

    return float(p0 - p1)


def predict(features, params):
    return run_quokka(build_circuit(features, params))
