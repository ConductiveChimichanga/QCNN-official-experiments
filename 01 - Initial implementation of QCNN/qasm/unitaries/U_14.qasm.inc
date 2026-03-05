//u_14 ansatz (6 parameters)
//based on: RY, RY, CRX, RY, RY, CRX
ry({{PFX_ry0}}) {{Q0}};
ry({{PFX_ry1}}) {{Q1}};

//crx(params[2], control=wires[1], target=wires[0])
ry(-pi/2) {{Q0}};
rz({{PFX_crx0}}/2) {{Q0}};
cx {{Q1}}, {{Q0}};
rz(-{{PFX_crx0}}/2) {{Q0}};
cx {{Q1}}, {{Q0}};
ry(pi/2) {{Q0}};

ry({{PFX_ry2}}) {{Q0}};
ry({{PFX_ry3}}) {{Q1}};

//crx(params[5], control=wires[0], target=wires[1])
ry(-pi/2) {{Q1}};
rz({{PFX_crx1}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
rz(-{{PFX_crx1}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
ry(pi/2) {{Q1}};
