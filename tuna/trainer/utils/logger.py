from pprint import pprint

class BaseLogger():
    def __init__(self, config):
        self.config = config

    def log_metric(self, metric):
        pprint(metric)

    def log(self, *args, **kwargs):
        print(*args, **kwargs)

class DisabledLogger(BaseLogger):
    def __init__(self, config):
        super().__init__(config)

    def log_metric(self, metric):
        pass

    def log(self, *args, **kwargs):
        pass

