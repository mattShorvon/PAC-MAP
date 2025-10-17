import pytest

from spn.structs import Variable
from spn.node.indicator import Indicator


@pytest.fixture(params=[(0, 2), (1, 3)])
def var(request):
    return Variable(request.param[0], request.param[1])


@pytest.fixture
def indicator(var):
    return Indicator(var.id, var.n_categories - 1)


class TestIndicator:
    def test_creation(self, var):
        with pytest.raises(TypeError):
            Indicator()
            Indicator(var)
            Indicator(var, -1)
            Indicator(var, var.n_categories + 1)
        assert Indicator(var, 0) is not None

    def test_var_id(self, var):
        ind = Indicator(var, 0)
        assert ind.var_id == var.id

    def test_variable(self, var):
        ind = Indicator(var, 0)
        assert ind.variable == var

    def test_assignment(self, var):
        ind = Indicator(var, 2)
        assert ind.assignment == 2
