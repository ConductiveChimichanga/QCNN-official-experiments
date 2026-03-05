//u_9 ansatz (2 parameters)
//based on: H, H, CZ, RX, RX
h {{Q0}};
h {{Q1}};

//cz decomposition: H-CX-H
h {{Q1}};
cx {{Q0}}, {{Q1}};
h {{Q1}};

rx({{PFX_rx0}}) {{Q0}};
rx({{PFX_rx1}}) {{Q1}};
