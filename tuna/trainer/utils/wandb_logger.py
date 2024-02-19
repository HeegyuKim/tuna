import wandb
import io
from .logger import BaseLogger
from pprint import pprint
import os


class WandbLogger(BaseLogger):

    def __init__(self, config):
        super().__init__(config)
        config_dict = config.__dict__
        config_dict["VM_NAME"] = os.environ.get("VM_NAME", "unknown")

        self.run = wandb.init(
            project=config.project,
            name=config.run_name.replace("/", "__").replace(",", "_"),
            config=config_dict
        )
    
    def define_eval_metrics(self, metrics):
        if metrics:
            for k, v in metrics.items():
                self.run.define_metric(f"eval/{k}", summary=v)

    def log_metric(self, metric):
        wandb.log(metric)
        pprint(metric)

    def log(self, *args, **kwargs):
        print(*args, **kwargs)
        # output = io.StringIO()
        # print(*args, file=output, **kwargs)
        # contents = output.getvalue()
        # output.close()
        