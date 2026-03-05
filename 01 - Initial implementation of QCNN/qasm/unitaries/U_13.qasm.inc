//u_13 ansatz (6 parameters)
//based on: RY, RY, CRZ, RY, RY, CRZ
ry({{PFX_ry0}}) {{Q0}};
ry({{PFX_ry1}}) {{Q1}};

//crz(params[2], control=wires[1], target=wires[0])
rz({{PFX_crz0}}/2) {{Q0}};
cx {{Q1}}, {{Q0}};
rz(-{{PFX_crz0}}/2) {{Q0}};
cx {{Q1}}, {{Q0}};

ry({{PFX_ry2}}) {{Q0}};
ry({{PFX_ry3}}) {{Q1}};

//crz(params[5], control=wires[0], target=wires[1])
rz({{PFX_crz1}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
rz(-{{PFX_crz1}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
