import torch.nn as nn
import torch.nn.functional as F
import torch

# CNN model (found with jax)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Spatial
        self.conv1 = nn.Conv2d(
            in_channels=4,
            out_channels=16,
            kernel_size=(2,2),
            stride=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=(2,2),
            stride=1
        )
        # Temporal
        self.dense1 = nn.Linear(
            in_features=4,
            out_features=64
        )
        # Distributional
        self.dense2 = nn.Linear(
            in_features=64+32*...*...,
            out_features=10)
        self.xi = nn.Parameter()
        
    def forward(self, x_s, x_t):
        # Spatial
        x_s = self.conv1(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        x_s = self.conv2(x_s)
        x_s = F.leaky_relu(x_s, 0.2)
        
        # Temporal
        x_t = self.dense1(x_t)
        
        # Distributional
        x_s = x_s.view(x_s.size(0), -1)
        x = torch.cat((x_s, x_t), dim=1)
        x = self.dense2(x)
        mu, sigma = x[:, :5], x[:, 5:]
        sigma = F.softplus(sigma)
        x = torch.cat((mu, sigma), dim=1)
        xi = F.sigmoid(self.xi)
        # Repeat xi to have the same batch size
        xi = xi.repeat(x.size(0), 1)
        x = torch.cat((x, xi), dim=1)
        return x
        
        
        

