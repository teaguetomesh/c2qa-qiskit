import numpy
from c2qa.operators import CVOperators
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
import numpy as np
from strawberryfields.backends.fockbackend import ops as sf_ops
import random


class TestUnitary:
    """Verify operators are unitary"""

    def setup_method(self, method):
        self.ops = CVOperators(4)

    def test_bs(self):
        assert is_unitary_matrix(self.ops.bs(random.random()))

    def test_d(self):
        assert is_unitary_matrix(self.ops.d(random.random()))

    def test_r(self):
        assert is_unitary_matrix(self.ops.r(random.random()))

    def test_s(self):
        assert is_unitary_matrix(self.ops.s(random.random()))

    def test_s2(self):
        assert is_unitary_matrix(self.ops.s2(random.random()))


class TestMatrices:
    """Test that the operators produce the values we expect.
    TODO - would be better to test against known input & output values vs simply non-zero
    """

    def setup_method(self, method):
        self.ops = CVOperators(4)

    def test_bs(self):
        one = self.ops.bs(1)
        rand = self.ops.bs(random.random())

        assert not numpy.allclose(one, rand)

    def test_d(self):
        one = self.ops.d(1)
        rand = self.ops.d(random.random())

        assert not numpy.allclose(one, rand)

    def test_r(self):
        one = self.ops.r(1)
        rand = self.ops.r(random.random())

        assert not numpy.allclose(one, rand)

    def test_s(self):
        one = self.ops.s(1)
        rand = self.ops.s(random.random())

        assert not numpy.allclose(one, rand)

    def test_s2(self):
        one = self.ops.s2(1)
        rand = self.ops.s2(random.random())

        assert not numpy.allclose(one, rand)
