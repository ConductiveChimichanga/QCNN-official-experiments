//u_so4 ansatz (6 parameters)
//based on: RY, RY, CNOT, RY, RY, CNOT, RY, RY
ry({{PFX_ry0}}) {{Q0}};
ry({{PFX_ry1}}) {{Q1}};
cx {{Q0}}, {{Q1}};
ry({{PFX_ry2}}) {{Q0}};
ry({{PFX_ry3}}) {{Q1}};
cx {{Q0}}, {{Q1}};
ry({{PFX_ry4}}) {{Q0}};
ry({{PFX_ry5}}) {{Q1}};
