//pooling ansatz
//two-qubit pooling operation with controlled rotations.
//parameters: crz_angle, crx_angle
//qubits: Q0, Q1

//crz decomposition
rz({{PFX_crz_angle}}/2) q[{{Q1}}];
cx q[{{Q0}}],q[{{Q1}}];
rz(-{{PFX_crz_angle}}/2) q[{{Q1}}];
cx q[{{Q0}}],q[{{Q1}}];

//pauli x on control
x q[{{Q0}}];

//crx decomposition via rotation to Z basis
rz(1.570796) q[{{Q1}}];
rx(1.570796) q[{{Q1}}];
rz(-1.570796) q[{{Q1}}];

rz({{PFX_crx_angle}}/2) q[{{Q1}}];
cx q[{{Q0}}],q[{{Q1}}];
rz(-{{PFX_crx_angle}}/2) q[{{Q1}}];
cx q[{{Q0}}],q[{{Q1}}];

rz(1.570796) q[{{Q1}}];
rx(-1.570796) q[{{Q1}}];
rz(-1.570796) q[{{Q1}}];
