#assembles a full qasm template from modular unitary files + architecture json

import json
import re
from pathlib import Path

UNITARY_PARAMS = {
    "conv_block": [
        "u3a_theta", "u3a_phi", "u3a_lambda",
        "u3b_theta", "u3b_phi", "u3b_lambda",
        "ry1", "rz1", "ry2",
        "u3c_theta", "u3c_phi", "u3c_lambda",
        "u3d_theta", "u3d_phi", "u3d_lambda",
    ],
    "simple_entangle": ["theta0", "phi0", "theta1", "phi1"],
    "pooling_ansatz": ["crz_angle", "crx_angle"],
}

def _load_unitaries(unitaries_dir):
    #read all *.qasm.inc files into {name: template_text}
    d = Path(unitaries_dir)
    if not d.exists():
        raise FileNotFoundError(f"unitaries dir not found: {d}")
    return {
        f.name.replace(".qasm.inc", ""): f.read_text(encoding="utf-8").strip()
        for f in d.glob("*.qasm.inc")
    }

def _extract_params(template):
    #find unique {{PFX_*}} parameter names in order
    seen, out = set(), []
    for m in re.findall(r"\{\{PFX_(\w+)\}\}", template):
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out

def _render_pair(prefix, q0, q1, template, params):
    #substitute qubit and param placeholders for one pair
    t = template.replace("{{Q0}}", f"q[{q0}]").replace("{{Q1}}", f"q[{q1}]")
    for p in params:
        t = t.replace(f"{{{{PFX_{p}}}}}", f"{{{{{prefix}_{p}}}}}")
    return t

def build_qasm_template(unitaries_dir, architecture_path, output_path):
    #main entry: unitaries + architecture -> full qasm template
    unitaries = _load_unitaries(unitaries_dir)
    arch = json.loads(Path(architecture_path).read_text(encoding="utf-8"))

    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{arch['n_qubits']}];",
        "",
    ]

    for i, layer in enumerate(arch["layers"], 1):
        name = layer.get("name", f"layer_{i}")
        prefix = layer["prefix"]
        unitary = layer.get("unitary", "conv_block")
        if unitary not in unitaries:
            raise ValueError(f"unitary '{unitary}' not found")

        tmpl = unitaries[unitary]
        params = UNITARY_PARAMS.get(unitary) or _extract_params(tmpl)

        lines.append(f"//{name}")
        for q0, q1 in layer["pairs"]:
            lines.append(f"//{unitary} on ({q0},{q1})")
            lines.append(_render_pair(prefix, q0, q1, tmpl, params))
            lines.append("")
        if layer.get("comment"):
            lines.append(f"//{layer['comment']}")
            lines.append("")

    Path(output_path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    build_qasm_template(
        str(root / "unitaries"),
        str(root / "architecture_simple.json"),
        str(root / "unitaries_simple.qasm"),
    )
    print("wrote unitaries_simple.qasm")
