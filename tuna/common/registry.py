
class Registry:
    def __init__(self, name: str = "registry"):
        self.name = name
        self._registry = {}

    def __call__(self, key):
        def inner_wrapper(wrapped_class):
            self._registry[key] = wrapped_class
            return wrapped_class
        return inner_wrapper
    
    def register(self, key):
        def inner_wrapper(wrapped_class):
            self._registry[key] = wrapped_class
            return wrapped_class
        return inner_wrapper

    def __getitem__(self, key):
        return self._registry.get(key)

    def get(self, key):
        return self._registry.get(key)
