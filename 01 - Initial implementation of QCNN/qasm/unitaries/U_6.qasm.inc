//u_6 ansatz (10 parameters)
//based on: RX, RX, RZ, RZ, CRX, CRX, RX, RX, RZ, RZ
rx({{PFX_rx0}}) {{Q0}};
rx({{PFX_rx1}}) {{Q1}};
rz({{PFX_rz0}}) {{Q0}};
rz({{PFX_rz1}}) {{Q1}};

//crx(params[4], control=wires[1], target=wires[0])
ry(-pi/2) {{Q0}};
rz({{PFX_crx0}}/2) {{Q0}};
cx {{Q1}}, {{Q0}};
rz(-{{PFX_crx0}}/2) {{Q0}};
cx {{Q1}}, {{Q0}};
ry(pi/2) {{Q0}};

//crx(params[5], control=wires[0], target=wires[1])
ry(-pi/2) {{Q1}};
rz({{PFX_crx1}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
rz(-{{PFX_crx1}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
ry(pi/2) {{Q1}};

rx({{PFX_rx2}}) {{Q0}};
rx({{PFX_rx3}}) {{Q1}};
rz({{PFX_rz2}}) {{Q0}};
rz({{PFX_rz3}}) {{Q1}};
