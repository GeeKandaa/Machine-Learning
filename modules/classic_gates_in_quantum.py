from qiskit import *
from qiskit.tools.visualization import plot_histogram
import numpy as np


## The following functions are quantum implementations of classical gates
# https://qiskit.org/textbook/ch-ex/ex1.html

def NOT(val):
    # Initialisation of registers and quantum circuit
    q = QuantumRegister(1)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q,c)

    # Circuitry
    if val == 1:
        qc.x(q[0])
    qc.x(q[0])
    qc.measure(q[0],c[0])

    # Run q. sim.
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1)
    ret = next(iter(job.result().get_counts()))

    return ret

def XOR(val0, val1):
    # Initialisation of registers and quantum circuit
    q = QuantumRegister(2)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q,c)

    # Circuitry
    if val0 == 1:
        qc.x(q[0])
    if val1 == 1:
        qc.x(q[1])
    qc.cx(q[0],q[1])
    qc.measure(q[1],c[0])
    
    # Run q. sim.
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1, memory=True)
    ret = job.result().get_memory()[0]

    return ret

def AND(val0, val1):
    # Initialisation of registers and quantum circuit
    q = QuantumRegister(3)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q,c)

    # Circuitry
    if val0 == 1:
        qc.x(q[0])
    if val1 == 1:
        qc.x(q[1])
    qc.ccx(q[0],q[1],q[2])
    qc.measure(q[2],c[0])
    
    # Run q. sim.
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1, memory=True)
    ret = job.result().get_memory()[0]

    return ret

def NAND(val0, val1):
    # Initialisation of registers and quantum circuit
    q = QuantumRegister(3)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q,c)

    # Circuitry
    if val0 == 1:
        qc.x(q[0])
    if val1 == 1:
        qc.x(q[1])
    qc.x(q[2])
    qc.ccx(q[0],q[1],q[2])
    qc.measure(q[2],c[0])
    
    # Run q. sim.
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1, memory=True)
    ret = job.result().get_memory()[0]

    return ret

def OR(val0, val1):
    # Initialisation of registers and quantum circuit
    q = QuantumRegister(3)
    c = ClassicalRegister(1)
    qc = QuantumCircuit(q,c)

    # Circuitry
    if val0 == 1:
        qc.x(q[0])
    if val1 == 1:
        qc.x(q[1])
    qc.ccx(q[0],q[1],q[2])
    qc.cx(q[0],q[1])
    qc.ccx(q[0],q[1],q[2])
    qc.cx(q[1],q[0])
    qc.ccx(q[0],q[1],q[2])
    qc.measure(q[2],c[0])
    
    # Run q. sim.
    backend = Aer.get_backend('qasm_simulator')
    job = execute(qc,backend,shots=1, memory=True)
    ret = job.result().get_memory()[0]

    return ret

# Check functionality (debug)
##print('\nNOT(0):',NOT(0))
##print('\nNOT(1):',NOT(1))
##print('\nXOR(0,0):',XOR(0,0))
##print('\nXOR(1,0):',XOR(1,0))
##print('\nXOR(0,1):',XOR(0,1))
##print('\nXOR(1,1):',XOR(1,1))
##print('\nAND(0,0):',AND(0,0))
##print('\nAND(1,0):',AND(1,0))
##print('\nAND(0,1):',AND(0,1))
##print('\nAND(1,1):',AND(1,1))
##print('\nNAND(0,0):',NAND(0,0))
##print('\nNAND(1,0):',NAND(1,0))
##print('\nNAND(0,1):',NAND(0,1))
##print('\nNAND(1,1):',NAND(1,1))
##print('\nOR(0,0):',OR(0,0))
##print('\nOR(1,0):',OR(1,0))
##print('\nOR(0,1):',OR(0,1))
##print('\nOR(1,1):',OR(1,1))
