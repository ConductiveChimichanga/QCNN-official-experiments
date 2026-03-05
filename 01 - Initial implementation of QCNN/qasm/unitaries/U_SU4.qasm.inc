//u_su4 ansatz (15 parameters)
//full SU(4) decomposition using U3 gates
//based on: U3, U3, CNOT, RY, RZ, CNOT, RY, CNOT, U3, U3
u3({{PFX_u3a_theta}}, {{PFX_u3a_phi}}, {{PFX_u3a_lambda}}) {{Q0}};
u3({{PFX_u3b_theta}}, {{PFX_u3b_phi}}, {{PFX_u3b_lambda}}) {{Q1}};
cx {{Q0}}, {{Q1}};
ry({{PFX_ry1}}) {{Q0}};
rz({{PFX_rz1}}) {{Q1}};
cx {{Q1}}, {{Q0}};
ry({{PFX_ry2}}) {{Q0}};
cx {{Q0}}, {{Q1}};
u3({{PFX_u3c_theta}}, {{PFX_u3c_phi}}, {{PFX_u3c_lambda}}) {{Q0}};
u3({{PFX_u3d_theta}}, {{PFX_u3d_phi}}, {{PFX_u3d_lambda}}) {{Q1}};
