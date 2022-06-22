import torch
import torch.nn as nn
import torch.nn.functional as F

from NF_base import *

class ExpMap(BaseNormalizingFlow):
    """
        Radial Exponential Map, introduced in [1]. See also [2] for extensions.
        
        Refs:
        [1] Rezende, Danilo Jimenez, et al. "Normalizing flows on tori and spheres." International Conference on Machine Learning. PMLR, 2020.
        [2] Cohen, Samuel, Brandon Amos, and Yaron Lipman. "Riemannian Convex Potential Maps." International Conference on Machine Learning. PMLR, 2021.
    """
    def __init__(self, dim, n_radial_components):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(n_radial_components, requires_grad=True) / n_radial_components)
        self.beta = nn.Parameter(torch.rand(n_radial_components, requires_grad=True))
        self.mus = nn.Parameter(torch.randn(n_radial_components, dim, requires_grad=True))
        
    def forward(self, x):
        # Enforce constraints of $\sum_k \alpha_k \leq 1; \forall \beta_k > 0$
        alpha = F.softmax(self.alpha, dim=-1)
        beta = F.softplus(self.beta)
        mus = F.normalize(self.mus, p=2, dim=-1)
        
        x.requires_grad_(True)
        phi = torch.sum((alpha/beta) * torch.exp(beta * (x@mus.T-1)), axis=-1)
#         grad_phi = torch.autograd.grad(phi.sum(), x)[0]
        grad_phi = (alpha * torch.exp(beta * (x@mus.T -1))) @ mus
        
        ## Projection on T_x S^d of \nabla \phi
        v = grad_phi -  torch.sum(x*grad_phi, axis=-1)[:,None]*x
        norm_v = torch.linalg.norm(v, dim=-1)[:,None]
        exp = x * torch.cos(norm_v) + (v/norm_v) * torch.sin(norm_v)

        ## Orthonormal basis T_x S^d        
        grad_phi_normalized = v/norm_v
        E = torch.dstack([grad_phi_normalized, torch.cross(x, grad_phi_normalized)])
                
        ## Compute Jf
        Jf = []
        for i in range(exp.shape[1]):
            Jf.append(torch.autograd.grad(exp[:, i].sum(), x, create_graph=True, retain_graph=True)[0])

        # Jf: (batch_size, d, d)
        Jf = torch.stack(Jf, dim=1)
                
        G = torch.matmul(Jf, E) 
        log_density = torch.logdet(torch.matmul(torch.transpose(G,1,2), G))/2
        
        return exp, log_density    
    
    def backward(self, z):
        pass
    
    
def create_NF(d=3, n_blocks=6, n_components=5):
    flows = []
    for k in range(n_blocks):
        radialBlock = ExpMap(d, n_components)
        flows.append(radialBlock)

    model = NormalizingFlows(flows).to(device)
    return model