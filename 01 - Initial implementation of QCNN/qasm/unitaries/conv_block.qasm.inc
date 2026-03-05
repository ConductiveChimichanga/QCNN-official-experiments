//qcnn convolution block
//two-qubit convolution operation with parametric rotations and entanglement.
//parameters: u3a_theta, u3a_phi, u3a_lambda, u3b_theta, u3b_phi, u3b_lambda,
//ry1, rz1, ry2, u3c_theta, u3c_phi, u3c_lambda, u3d_theta, u3d_phi, u3d_lambda
//qubits: Q0, Q1

rz({{PFX_u3a_lambda}}) q[{{Q0}}];
ry({{PFX_u3a_theta}}) q[{{Q0}}];
rz({{PFX_u3a_phi}}) q[{{Q0}}];

rz({{PFX_u3b_lambda}}) q[{{Q1}}];
ry({{PFX_u3b_theta}}) q[{{Q1}}];
rz({{PFX_u3b_phi}}) q[{{Q1}}];

cx q[{{Q1}}],q[{{Q0}}];
ry({{PFX_ry1}}) q[{{Q0}}];
rz({{PFX_rz1}}) q[{{Q1}}];
cx q[{{Q0}}],q[{{Q1}}];
ry({{PFX_ry2}}) q[{{Q0}}];
cx q[{{Q1}}],q[{{Q0}}];

rz({{PFX_u3c_lambda}}) q[{{Q0}}];
ry({{PFX_u3c_theta}}) q[{{Q0}}];
rz({{PFX_u3c_phi}}) q[{{Q0}}];

rz({{PFX_u3d_lambda}}) q[{{Q1}}];
ry({{PFX_u3d_theta}}) q[{{Q1}}];
rz({{PFX_u3d_phi}}) q[{{Q1}}];
