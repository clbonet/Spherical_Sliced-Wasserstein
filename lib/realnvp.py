import torch

from NF_base import *


class shifting(nn.Module):
    def __init__(self, d_in, nh, d_out, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_in,nh))
        for i in range(n_layers):
            self.layers.append(nn.Linear(nh,nh))
        self.layers.append(nn.Linear(nh,d_out))

    def forward(self, x):
        for layer in self.layers:
            x = F.leaky_relu(layer(x),0.2)
        return x


class scaling(nn.Module):
    def __init__(self, d_in, nh, d_out, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(d_in,nh))
        for i in range(n_layers):
            self.layers.append(nn.Linear(nh,nh))
        self.layers.append(nn.Linear(nh,d_out))

    def forward(self, x):
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return x


def create_RealNVP(nh=16,nl=5,d=2):

    shiftings = [shifting(d//2,nh,(d+1)//2,2) for k in range(nl)]
    scalings = [shifting(d//2,nh,(d+1)//2,2) for k in range(nl)]

    flows = []
    for i in range(nl):
        flows.append(AffineCoupling(scalings[i],shiftings[i],d))
        flows.append(Reverse(d))

    model = NormalizingFlows(flows).to(device)
    return model