import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from mypythonlib.config import settings
from mypythonlib.helper import softmax_with_mask

class BaseAgent(ABC):
    @abstractmethod
    def forward(self, x, mask):
        raise NotImplementedError

    @abstractmethod
    def save(self, filename):
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def load(cls, filename):
        raise NotImplementedError

class MyModel(nn.Module, BaseAgent):
    def __init__(self, input_size, output_size, hiddenlayers=None):
        super().__init__()

        if not hiddenlayers:
            hiddenlayers = [512, 256, 128]
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hiddenlayers[0]))
        for i in range(1, len(hiddenlayers)):
            self.layers.append(nn.Linear(hiddenlayers[i-1], hiddenlayers[i]))
        self.output_layer = nn.Linear(hiddenlayers[-1], output_size)

        self.config = {"input_size": input_size, "output_size": output_size, "hiddenlayers": hiddenlayers}

    def forward(self, x, mask):
        for layer in self.layers:
            x = F.relu(layer(x))

        logits = self.output_layer(x)
        return softmax_with_mask(logits, mask)

    def save(self, filename=None):
        if filename is None:
            filename = f"{__class__.__name__}.pth"
        path = settings.models_path / filename
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, filename=None):
        if filename is None:
            filename = f"{__class__.__name__}.pth"
        path = settings.models_path / filename
        checkpoint = torch.load(path, weights_only=False)

        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

if __name__ == "__main__":
    model = MyModel(input_size=16, output_size=32)
    filename = "test.pth"
    model.save(filename)
    print(f"model.config: {model.config}")
    model2 = MyModel.load(filename)
    print(f"model2.config: {model2.config}")
