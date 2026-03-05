//u_5 ansatz (10 parameters)
//based on: RX, RX, RZ, RZ, CRZ, CRZ, RX, RX, RZ, RZ
rx({{PFX_rx0}}) {{Q0}};
rx({{PFX_rx1}}) {{Q1}};
rz({{PFX_rz0}}) {{Q0}};
rz({{PFX_rz1}}) {{Q1}};

//crz(params[4], control=wires[1], target=wires[0])
rz({{PFX_crz0}}/2) {{Q0}};
cx {{Q1}}, {{Q0}};
rz(-{{PFX_crz0}}/2) {{Q0}};
cx {{Q1}}, {{Q0}};

//crz(params[5], control=wires[0], target=wires[1])
rz({{PFX_crz1}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
rz(-{{PFX_crz1}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};

rx({{PFX_rx2}}) {{Q0}};
rx({{PFX_rx3}}) {{Q1}};
rz({{PFX_rz2}}) {{Q0}};
rz({{PFX_rz3}}) {{Q1}};
