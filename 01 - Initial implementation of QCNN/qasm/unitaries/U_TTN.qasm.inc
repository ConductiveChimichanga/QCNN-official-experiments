//u_ttn: Tree Tensor Network ansatz (2 parameters)
//based on: RY, RY, CNOT
ry({{PFX_ry0}}) {{Q0}};
ry({{PFX_ry1}}) {{Q1}};
cx {{Q0}}, {{Q1}};
