//pooling_ansatz1 (2 parameters)
//based on: CRZ, PauliX, CRX

//crz decomposition
rz({{PFX_crz_angle}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
rz(-{{PFX_crz_angle}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};

//pauli x
x {{Q0}};

//crx decomposition
ry(-pi/2) {{Q1}};
rz({{PFX_crx_angle}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
rz(-{{PFX_crx_angle}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
ry(pi/2) {{Q1}};
