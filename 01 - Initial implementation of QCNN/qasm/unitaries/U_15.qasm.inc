//u_15 ansatz (4 parameters)
//based on: RY, RY, CNOT, RY, RY, CNOT
ry({{PFX_ry0}}) {{Q0}};
ry({{PFX_ry1}}) {{Q1}};
cx {{Q1}}, {{Q0}};
ry({{PFX_ry2}}) {{Q0}};
ry({{PFX_ry3}}) {{Q1}};
cx {{Q0}}, {{Q1}};
