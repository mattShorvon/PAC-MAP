"""Actions are the preferred way of separating all the possible computations that
can be done by this program. Ideally, these functionalities should be an independent
program on their own, but their inputs (and sometimes outputs) are intrinsically
connected, so building a list of actions to be done in sequence -- which is
accomplished in the config/spn.json file -- is the preferred way of testing these
prototypes."""

from typing import Any, List, Mapping


class Action:
    necessary_params: List[str] = []
    key = 'None'

    def __init__(self, params: Mapping[str, Any]):
        self.params = params
        self._check_params()
        self.execute()

    def _check_params(self):
        for param in self.__class__.necessary_params:
            assert param in self.params, "Action '{}' needs parameter '{}'".format(self.__class__.key, param)

    def execute(self):
        raise NotImplementedError
