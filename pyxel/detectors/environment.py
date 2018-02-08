
class Environment:

    def __init__(self, temperature: float = None) -> None:
        self.temperature = temperature          # unit: K

    def __getstate__(self):
        return {'temperature': self.temperature}

    # TODO: create unittests for this method
    def __eq__(self, obj):
        assert isinstance(obj, Environment)
        return self.__getstate__() == obj.__getstate__()
