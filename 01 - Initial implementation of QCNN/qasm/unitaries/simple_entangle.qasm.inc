//simple entangling layer
//simpler two-qubit operation with basic rotations and one entangling gate.
//parameters: theta0, phi0, theta1, phi1
//qubits: Q0, Q1

ry({{PFX_theta0}}) q[{{Q0}}];
rz({{PFX_phi0}}) q[{{Q0}}];

ry({{PFX_theta1}}) q[{{Q1}}];
rz({{PFX_phi1}}) q[{{Q1}}];

cx q[{{Q0}}],q[{{Q1}}];
