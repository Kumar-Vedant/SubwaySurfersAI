import torch
from torch import nn

class BehaviorCloningModel(nn.Module):
    def __init__(self, agent_nn):
        super().__init__()
        self.agent_nn = agent_nn

    def forward(self, x):
        # normalize image input
        x = x / 255.0
        # shared CNN and FC layers
        x = self.agent_nn.network(x)

        # only actor output
        logits = self.agent_nn.actor(x)
        return logits
