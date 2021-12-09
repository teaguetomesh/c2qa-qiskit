from pprint import pprint
import sqlite3


import c2qa
from prettytable import from_db_cursor
import pytest
import qiskit


class TestMemoryUsage():

    def __cat_state(self, circuit: c2qa.CVCircuit, qmr: c2qa.QumodeRegister, qr: qiskit.QuantumRegister, cr: qiskit.ClassicalRegister, dist: float = 1.0):
        circuit.initialize([1, 0], qr[0])
        circuit.cv_initialize(0, qmr[0])

        circuit.h(qr[0])
        circuit.cv_cnd_d(dist, -dist, qr[0], qmr[0])
        circuit.h(qr[0])
        circuit.measure(qr[0], cr[0])


    @pytest.mark.parametrize("loops", range(1, 11))
    def test_memory_usage(self, loops):
        registers = []
        for _ in range(loops):
            registers.append(c2qa.QumodeRegister(num_qumodes=1, num_qubits_per_qumode=2))
            registers.append(qiskit.QuantumRegister(size=1))
            registers.append(qiskit.ClassicalRegister(size=1))
        
        circuit = c2qa.CVCircuit(*registers)

        for i in range(0, loops, 3):
            self.__cat_state(circuit, registers[i], registers[i+1], registers[i+2])
        
        c2qa.util.simulate(circuit)
       

    def teardown_class(self):
        con = sqlite3.connect(".pymon")
        cur = con.execute("SELECT ITEM_VARIANT, TOTAL_TIME, CPU_USAGE, MEM_USAGE FROM TEST_METRICS")
        mytable = from_db_cursor(cur)
        print(mytable)
