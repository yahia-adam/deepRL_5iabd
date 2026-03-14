from torch.utils.tensorboard import SummaryWriter
from mypythonlib.tracking.base_logger import BaseLogger

class TensorBoardLogger(BaseLogger):
    def __init__(self, log_dir: str, experiment_name: str):
        full_path = f"{log_dir}/{experiment_name}"
        self.writer = SummaryWriter(log_dir=full_path)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_dict(self, metrics: dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
