import c2qa
from c2qa.operators import CVOperators
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
import numpy as np
from strawberryfields.backends.fockbackend import ops as sf_ops

class TestUnitary:

    def setup_method(self, method):
        self.qmr = c2qa.QumodeRegister(1, 1)
        self.ops = CVOperators(self.qmr)

    def test_sf_d(self):
        op = self.ops.d(1)
        print(type(op))
        print(op)
        assert is_unitary_matrix(op), "QisKit must think FockWits is unitary"

        op = sf_ops.displacement(1, 0, self.qmr.cutoff)
        print(type(op))
        print(op)
        assert is_unitary_matrix(op), "QisKit must think Strawberry Fields is unitary"

    def test_bs(self):
        op = self.ops.bs(1)
        mat = np.array(op)
        # Compute A^dagger.A and see if it is identity matrix
        ident = np.conj(mat.T).dot(mat)
        unitary = np.allclose(np.eye(mat.shape[0]), mat.H * mat)
        unitary2 =  np.allclose(np.eye(len(op)), op.dot(op.T.conj()))
        assert is_unitary_matrix(op)

    def test_d(self):
        assert is_unitary_matrix(self.ops.d(1))

    def test_r(self):
        assert is_unitary_matrix(self.ops.r(1))

    def test_s(self):
        assert is_unitary_matrix(self.ops.s(1))

    def test_s2(self):
        assert is_unitary_matrix(self.ops.s2(1))
