from spn.actions.base import Action

ACTIONS = { cls.key: cls for cls in Action.__subclasses__() }
