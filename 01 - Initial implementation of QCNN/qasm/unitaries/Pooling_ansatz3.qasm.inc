//pooling_ansatz3 (3 parameters)
//based on: CRot(phi, theta, omega) - controlled rotation with 3 Euler angles

//crot decomposition: CRZ(phi) - CRY(theta) - CRZ(omega)
//crz(phi)
rz({{PFX_phi}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
rz(-{{PFX_phi}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};

//cry(theta)
ry({{PFX_theta}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
ry(-{{PFX_theta}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};

//crz(omega)
rz({{PFX_omega}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
rz(-{{PFX_omega}}/2) {{Q1}};
cx {{Q0}}, {{Q1}};
