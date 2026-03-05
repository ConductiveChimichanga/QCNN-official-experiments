OPENQASM 2.0;
include "qelib1.inc";
qreg q[4];

//conv1
//U_9 on (0,1)
//u_9 ansatz (2 parameters)
//based on: H, H, CZ, RX, RX
h q[0];
h q[1];

//cz decomposition: H-CX-H
h q[1];
cx q[0], q[1];
h q[1];

rx({{c1_rx0}}) q[0];
rx({{c1_rx1}}) q[1];

//U_9 on (2,3)
//u_9 ansatz (2 parameters)
//based on: H, H, CZ, RX, RX
h q[2];
h q[3];

//cz decomposition: H-CX-H
h q[3];
cx q[2], q[3];
h q[3];

rx({{c1_rx0}}) q[2];
rx({{c1_rx1}}) q[3];

//First convolutional layer on pairs (0,1) and (2,3)

//pool1
//Pooling_ansatz1 on (0,1)
//pooling_ansatz1 (2 parameters)
//based on: CRZ, PauliX, CRX

//crz decomposition
rz({{p1_crz_angle}}/2) q[1];
cx q[0], q[1];
rz(-{{p1_crz_angle}}/2) q[1];
cx q[0], q[1];

//pauli x
x q[0];

//crx decomposition
ry(-pi/2) q[1];
rz({{p1_crx_angle}}/2) q[1];
cx q[0], q[1];
rz(-{{p1_crx_angle}}/2) q[1];
cx q[0], q[1];
ry(pi/2) q[1];

//Pooling_ansatz1 on (2,3)
//pooling_ansatz1 (2 parameters)
//based on: CRZ, PauliX, CRX

//crz decomposition
rz({{p1_crz_angle}}/2) q[3];
cx q[2], q[3];
rz(-{{p1_crz_angle}}/2) q[3];
cx q[2], q[3];

//pauli x
x q[2];

//crx decomposition
ry(-pi/2) q[3];
rz({{p1_crx_angle}}/2) q[3];
cx q[2], q[3];
rz(-{{p1_crx_angle}}/2) q[3];
cx q[2], q[3];
ry(pi/2) q[3];

//First pooling layer reducing 4->2 qubits

//conv2
//U_9 on (0,2)
//u_9 ansatz (2 parameters)
//based on: H, H, CZ, RX, RX
h q[0];
h q[2];

//cz decomposition: H-CX-H
h q[2];
cx q[0], q[2];
h q[2];

rx({{c2_rx0}}) q[0];
rx({{c2_rx1}}) q[2];

//Second convolutional layer on remaining qubits

//pool2
//Pooling_ansatz1 on (0,2)
//pooling_ansatz1 (2 parameters)
//based on: CRZ, PauliX, CRX

//crz decomposition
rz({{p2_crz_angle}}/2) q[2];
cx q[0], q[2];
rz(-{{p2_crz_angle}}/2) q[2];
cx q[0], q[2];

//pauli x
x q[0];

//crx decomposition
ry(-pi/2) q[2];
rz({{p2_crx_angle}}/2) q[2];
cx q[0], q[2];
rz(-{{p2_crx_angle}}/2) q[2];
cx q[0], q[2];
ry(pi/2) q[2];

//Final pooling layer reducing to single qubit
