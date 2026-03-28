import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from deeprl_5iabd.config import settings
from deeprl_5iabd.helper import softmax_with_mask
from deeprl_5iabd.agents.base_agent import BaseAgent

class PolicyNetwork(nn.Module, BaseAgent):
    def __init__(self, name, input_size, output_size, hiddenlayers=None):
        super().__init__()
        self.name = name

        if hiddenlayers is None or len(hiddenlayers) == 0:
            hiddenlayers = [512, 256, 128]

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hiddenlayers[0]))
        for i in range(1, len(hiddenlayers)):
            self.layers.append(nn.Linear(hiddenlayers[i-1], hiddenlayers[i]))
        self.output_layer = nn.Linear(hiddenlayers[-1], output_size)

        self.config = {"name": self.name, "input_size": input_size, "output_size": output_size, "hiddenlayers": hiddenlayers}

    def forward(self, x, mask):
        for layer in self.layers:
            x = F.relu(layer(x))

        logits = self.output_layer(x)
        return softmax_with_mask(logits, mask)

    def save(self, filename=None):
        if filename is None:
            filename = f"{self.name}.pth"
        path = settings.models_path / filename
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config
        }
        torch.save(checkpoint, path)

    def clone(self, name=None):
        new_name = name if name else f"{self.name}_copy"
        cloned_model = PolicyNetwork(
            name=new_name, 
            input_size=self.config["input_size"], 
            output_size=self.config["output_size"], 
            hiddenlayers=self.config["hiddenlayers"]
        )
        
        cloned_model.load_state_dict(copy.deepcopy(self.state_dict()))
        cloned_model.eval() 
        return cloned_model

    @classmethod
    def load(cls, filename):
        path = settings.models_path / filename

        checkpoint = torch.load(path, weights_only=False)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model

if __name__ == "__main__":
    model = PolicyNetwork(name="test", input_size=16, output_size=32)
    filename = "test.pth"
    model.save(filename)
    print(f"model.config: {model.config}")
    model2 = PolicyNetwork.load(filename)
    print(f"model2.config: {model2.config}")
