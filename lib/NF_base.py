from abc import ABC, abstractmethod

import numpy as np
import scipy as sp

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform, TransformedDistribution, SigmoidTransform


device = "cuda" if torch.cuda.is_available() else "cpu"


class BaseNormalizingFlow(ABC, nn.Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def backward(self, z):
        pass


class AdditiveCoupling(BaseNormalizingFlow):
    """
    	Additive coupling layer

    	Refs:
    	- NICE: Non-linear independent components estimation
    """

    def __init__(self, coupling, dim):
        super().__init__()
        self.k = dim//2
        self.coupling = coupling

    def forward(self, x):
        x0, x1 = x[:,:self.k], x[:,self.k:]
        
        m = self.coupling(x0)
        z0 = x0
        z1 = x1+m
            
        z = torch.cat([z0,z1], dim=1)
        return z,torch.zeros(x.shape[0],device=device)
    
    def backward(self, z):
        z0, z1 = z[:,:self.k], z[:,self.k:]

        m = self.coupling(z0)
        x0 = z0
        x1 = z1-m

        x = torch.cat([x0,x1], dim=1)
        return x #, torch.zeros(z.shape[0],device=device)


class Scale(BaseNormalizingFlow):
    """
    	Scaling layer

    	Refs:
    	- NICE: Non-linear independent components estimation
    """

    def __init__(self, dim):
        super().__init__()
        self.log_s = nn.Parameter(torch.randn(1, dim, requires_grad=True))

    def forward(self, x):
        return torch.exp(self.log_s)*x, torch.sum(self.log_s, dim=1)
    
    def backward(self, z):
        return torch.exp(-self.log_s)*z #, -torch.sum(self.log_s, dim=1)


class AffineCoupling(BaseNormalizingFlow):
    """
    	Affine Coupling layer

    	Refs:
        - Density estimation using RealNVP
    """

    def __init__(self, scaling, shifting, dim):
        super().__init__()
        self.scaling = scaling
        self.shifting = shifting
        self.k = dim//2

    def forward(self, x):
        x0, x1 = x[:,:self.k], x[:,self.k:]

        s = self.scaling(x0)
        t = self.shifting(x0)
        z0 = x0
        z1 = torch.exp(s)*x1+t

        z = torch.cat([z0,z1], dim=1)
        return z, torch.sum(s, dim=1)


    def backward(self, z):
        z0, z1 = z[:,:self.k], z[:,self.k:]

        s = self.scaling(z0)
        t = self.shifting(z0)
        x0 = z0
        x1 = torch.exp(-s)*(z1-t)
        
        x = torch.cat([x0,x1], dim=1)
        return x #, -torch.sum(s, dim=1)

class Reverse(BaseNormalizingFlow):
    """
    	Reverse the indices

        Refs:
        - https://github.com/acids-ircam/pytorch_flows/blob/master/flows_04.ipynb
    """

    def __init__(self, dim):
        super().__init__()
        self.permute = torch.arange(dim-1, -1, -1)
        self.inverse = torch.argsort(self.permute)

    def forward(self, x):
        return x[:, self.permute], torch.zeros(x.size(0), device=device)

    def backward(self, z):
        return z[:, self.inverse] #, torch.zeros(z.size(0), device=device)


class Shuffle(Reverse):
    """
        Apply a random permutation of the indices
    """
    def __init__(self, d):
        super().__init__(d)
        self.permute = torch.randperm(d)
        self.inverse = torch.argsort(self.permute)
        

class BatchNorm(BaseNormalizingFlow):
    """
        Refs: 
        - https://github.com/acids-ircam/pytorch_flows/blob/master/flows_04.ipynb
        - Masked Autoregressive Flows for Density Estimation 
        - Density Estimation Using Real NVP
    """
    def __init__(self, dim, eps=1e-5, momentum=0.95):
        super().__init__()
        self.eps = eps
        self.momentum = momentum ## To compute train set mean
        self.train_mean = torch.zeros(dim, device=device)
        self.train_var = torch.ones(dim, device=device)

        self.gamma = nn.Parameter(torch.ones(dim, requires_grad=True))
        self.beta = nn.Parameter(torch.ones(dim, requires_grad=True))

    def forward(self, x):
        """
            mean=batch_mean in training time, mean of the entire dataset in test
        """
        if self.training:
            self.batch_mean = x.mean(0)
            self.batch_var = (x-x.mean(0)).pow(2).mean(0)+self.eps

            self.train_mean = self.momentum*self.train_mean+(1-self.momentum)*self.batch_mean
            self.train_var = self.momentum*self.train_var+(1-self.momentum)*self.batch_var

            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.train_mean
            var = self.train_var

        z = torch.exp(self.gamma)*(x-mean)/var.sqrt()+self.beta
        log_det = torch.sum(self.gamma-0.5*torch.log(var))
        return z, log_det

    def backward(self, z):
        if self.training:
            mean = self.batch_mean
            var = self.batch_var
        else:
            mean = self.train_mean
            var = self.train_var

        x = (z-self.beta)*torch.exp(-self.gamma)*var.sqrt()+mean
        log_det = torch.sum(-self.gamma+0.5*torch.log(var))
        return x #, log_det
        
        
class LUInvertible(BaseNormalizingFlow):
    """
    	Invertible 1x1 convolution (based on LU decomposition)
    	
    	Refs:
    	- Glow: Generative flow with invertible 1×1 convolutions
        - https://github.com/ikostrikov/pytorch-flows/blob/master/flows.py
        - https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.W = torch.Tensor(dim, dim)
        nn.init.orthogonal_(self.W)

        # P, L, U = torch.lu_unpack(*self.W.lu())
        P, L, U = sp.linalg.lu(self.W.numpy())
        self.P = torch.from_numpy(P)
        self.L = nn.Parameter(torch.from_numpy(L))
        self.S = nn.Parameter(torch.from_numpy(U).diag())
        self.U = nn.Parameter(torch.triu(torch.from_numpy(U),1))

    def forward(self, x):
        P = self.P.to(device)
        L = torch.tril(self.L,-1)+torch.diag(torch.ones(self.dim,device=device))
        U = torch.triu(self.U,1)+torch.diag(self.S)
        W = P @ L @ U
        return x@W, torch.sum(torch.log(torch.abs(self.S)))

    def backward(self, z):
        P = self.P.to(device)
        L = torch.tril(self.L,-1)+torch.diag(torch.ones(self.dim, device=device))
        U = torch.triu(self.U,1)+torch.diag(self.S)
        W = P @ L @ U
        return z@torch.inverse(W) #, -torch.sum(torch.log(torch.abs(self.S)))


class PlanarFlow(BaseNormalizingFlow):
    """
    	Refs
        - Variational Inference with Normalizing Flows, https://arxiv.org/pdf/1505.05770.pdf
    """
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(1, dim, requires_grad=True))
        self.w = nn.Parameter(torch.randn(1, dim, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, 1, requires_grad=True))

    def forward(self, x):
        # enforce invertibility
        wu = self.w@self.u.t()
        m_wu = -1+torch.log(1+torch.exp(wu))
        u_hat = self.u+(m_wu-wu)*self.w/torch.sum(self.w**2)

        z = x+u_hat*torch.tanh(x@self.w.t()+self.b)
        psi = (1-torch.pow(torch.tanh(x@self.w.t()+self.b),2))*self.w
        log_det = torch.log(1+psi@u_hat.t())
        return z, log_det[:,0]

    def backward(self, z):
        # can't compute it analytically
        return NotImplementedError
        
        
class RadialFlow(BaseNormalizingFlow):
    """
    	Refs
        - Variational Inference with Normalizing Flows, https://arxiv.org/pdf/1505.05770.pdf
    """
    def __init__(self, dim):
        super().__init__()
        self.d = dim
        self.log_alpha = nn.Parameter(torch.randn(1, requires_grad=True))
        self.beta = nn.Parameter(torch.randn(1, requires_grad=True))
        self.z0 = nn.Parameter(torch.randn(dim, requires_grad=True))   

    def forward(self, x):
        r = torch.norm(x-self.z0,dim=-1,keepdim=True)

        alpha = torch.exp(self.log_alpha)
        h = 1/(alpha+r)
        beta = -alpha+torch.log(1+torch.exp(self.beta))

        z = x+beta*h*(x-self.z0)
        
        log_det = (self.d-1)*torch.log(1+beta*h)+torch.log(1+beta*h-beta*r/(alpha+r)**2)

        return z, log_det[:,0]
    
    def backward(self, z):
        raise NotImplementerError
    
    
class AffineConstantFlow(BaseNormalizingFlow):
    """ 
    	Scales + Shifts the flow by (learned) constants per dimension.
    	In NICE paper there is a Scaling layer which is a special case of this where t is None

		Refs:
	    - https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    """
    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None
        
    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        log_det = torch.sum(s, dim=1)
        return z, log_det
    
    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        log_det = torch.sum(-s, dim=1)
        return x #, log_det


class ActNorm(AffineConstantFlow):
    """
    	Really an AffineConstantFlow but with a data-dependent initialization,
	    where on the very first batch we clever initialize the s,t so that the output
	    is unit gaussian. As described in Glow paper.
		
		Refs:
    	- https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    	- Glow: Generative flow with invertible 1×1 convolutions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_dep_init_done = False
    
    def forward(self, x):
        # first batch is used for init
        if not self.data_dep_init_done:
            assert self.s is not None and self.t is not None # for now
            self.s.data = (-torch.log(x.std(dim=0, keepdim=True))).detach()
            self.t.data = (-(x * torch.exp(self.s)).mean(dim=0, keepdim=True)).detach()
            self.data_dep_init_done = True
        return super().forward(x)
        

class NormalizingFlows(BaseNormalizingFlow):
    """
        Composition of flows
        
        Refs: 
        - https://github.com/karpathy/pytorch-normalizing-flows/blob/master/nflib/flows.py
    """
    def __init__(self, flows):
        """
    		Inputs:
    		- flows: list of BaseNormalizingFlows objects
    	"""
        super().__init__()
        self.flows = nn.ModuleList(flows)
        
    def forward(self, x):
        log_det = torch.zeros(x.shape[0], device=device)
        zs = [x]
        for flow in self.flows:
            x, log_det_i = flow(x)
            log_det += log_det_i
            zs.append(x)
        return zs, log_det
    
    def backward(self, z):
        log_det = torch.zeros(z.shape[0], device=device)
        xs = [z]
        for flow in self.flows[::-1]:
            z = flow.backward(z)
#             log_det += log_det_i
            xs.append(z)
        return xs #, log_det
        
