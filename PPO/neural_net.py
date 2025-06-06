import torch
from torch import nn
import numpy as np
from torch.distributions import Categorical

class AgentNN(nn.Module):
    def __init__(self, input_shape, n_actions, freeze=False):
        super().__init__()

        # add 3 convolutional layers with ReLU activation
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # find the correct dimensions dynamically
        conv_out_size = self._get_conv_out(input_shape)

        # add a flatten and a linear layer
        self.network = nn.Sequential(
            self.conv_layers,
            nn.Flatten(),
            nn.Linear(conv_out_size, 256),
            nn.ReLU(),
        )

        # actor and critic heads
        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)
        
        # if static target network, don't update network parameters
        if freeze:
            self._freeze()

        # set the training device to gpu, if available, and cpu as a fallback
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, x):
        x = x / 255.0
        x = self.network(x)

        # get the probability distribution over actions using actor
        dist = Categorical(logits=self.actor(x))
        # get value of this state using critic
        value = self.critic(x)
        return dist, value

    # calculate the length of the output of the convolutional layers by passing a dummy tensor through it
    def _get_conv_out(self, shape):
        # create a dummy tensor with all zeros and pass it through convolutional layers
        o = self.conv_layers(torch.zeros(1, *shape))
        # calculate the total no. by multiplying all dimensions
        return int(np.prod(o.size()))

    # if target network, don't update any parameters
    def _freeze(self):
        for p in self.network.parameters():
            p.requires_grad = False