import sys
import torch
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from tqdm.auto import trange
from copy import deepcopy

sys.path.append("../lib")

from sw_sphere import sliced_wasserstein_sphere
from swae.sw import sliced_wasserstein
from power_spherical import *
from exp_map_nf import create_NF
from utils_sphere import *

parser = argparse.ArgumentParser()
parser.add_argument("--sw_sphere", help="If true, use sw_sphere", action="store_true")
parser.add_argument("--ntry", type=int, default=10, help="number of restart")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

def ULA_sphere(V, n_particles=1000, d=2, dt=1e-3, n_steps=4000, device=device, 
               init_distr=None, init_particles=None, bar=False):
    normal = D.MultivariateNormal(torch.zeros(d, device=device),torch.eye(d, device=device))

    if init_particles is None:
        x0 = normal.sample((n_particles,))
        x0 = F.normalize(x0, p=2, dim=-1) ## unif on sphere
    else:
        x0 = init_particles
        
    xk = x0.clone()
    
    L = [x0.clone().detach().cpu()]
    
    if bar:
        pbar = trange(n_steps)
    else:
        pbar = range(n_steps)
    
    for k in pbar:
        xk.requires_grad_(True)
        grad_V = torch.autograd.grad(V(xk).sum(), xk)[0]
        W = normal.sample((n_particles,))
        xk = xk.detach()
        
        v = -grad_V*dt+np.sqrt(2*dt)*W
        v = v - torch.sum(v*xk, axis=-1)[:,None] * xk ## projection on the Tangent space
        norm_v = torch.linalg.norm(v, axis=-1)[:,None]

        xk = xk*torch.cos(norm_v) + torch.sin(norm_v) * v/norm_v

        
        L.append(xk.clone().detach().cpu())
    
    return xk, L


def kl_ess(log_model_prob, target_prob):
    weights = target_prob / np.exp(log_model_prob)
    Z = np.mean(weights)
    KL = np.mean(log_model_prob - np.log(target_prob)) + np.log(Z)
    ESS = np.sum(weights) ** 2 / np.sum(weights ** 2)
    return Z, KL, ESS


def swvi_nf(n_epochs, V, lr, model=None, d=3, n_particles=100, steps_mcmc=20, 
            dt_mcmc=1e-3, n_projs=1000, device=device, plot=False):
    pbar = trange(n_epochs)
    
    L = []
    
    if model is None:
        model = create_NF(d).to(device)
        
    L.append(deepcopy(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for e in pbar:
        optimizer.zero_grad()
        
        noise = F.normalize(torch.randn((n_particles,d), device=device), p=2, dim=-1)
        z0, _ = model(noise)
        zt, _ = ULA_sphere(V, device=device, init_particles=z0[-1], d=d,
                           n_steps=steps_mcmc, n_particles=z0[-1].shape[0], dt=dt_mcmc)
        
        ## grad descent on sw
        if args.sw_sphere:
            sw = sliced_wasserstein_sphere(z0[-1], zt, num_projections=n_projs, device=device, p=2)/2
        else:
            sw = sliced_wasserstein(z0[-1], zt, num_projections=n_projs, device=device, p=2)/2
        sw.backward()
        optimizer.step()
        
            
        if plot and e%100 == 0:
            print("k="+str(e))
            target = lambda x : np.exp(-V(torch.tensor(x, dtype=torch.float, device=device)).cpu().numpy())
            
            noise = F.normalize(torch.randn((n_particles,d), device=device), p=2, dim=-1)
            z0, _ = model(noise)
            
            scatter_mollweide(z0[-1].detach().cpu(), target)
            scatter_mollweide(zt.detach().cpu(), target)
            
            kernel = gaussian_kde(z0[-1].T.detach().cpu())
            plot_target_density(lambda x: kernel.pdf(x.T))
            
        if e%100 == 0:
            L.append(deepcopy(model))
                
    return model, L


mus = torch.tensor([[1.5, 0.7+np.pi/2], [1, -1+np.pi/2], [5, 0.6+np.pi/2], [4, -0.7+np.pi/2]], device=device)
target_mus = spherical_to_euclidean_torch(mus)

def target_density(x):
    m = torch.matmul(x, target_mus.T)
    return torch.sum(torch.exp(10 * m), dim=-1)

target_mu = spherical_to_euclidean(np.array([
    [1.5, 0.7 + np.pi / 2],
    [1., -1. + np.pi / 2],
    [5., 0.6 + np.pi / 2],  # 0.5 -> 5.!
    [4., -0.7 + np.pi / 2]
]))

def s2_target(x):
    xe = np.dot(x, target_mu.T)
    return np.sum(np.exp(10 * xe), axis=1)


if __name__ == "__main__":    
    V = lambda x: -torch.log(target_density(x))
    
    L_kl = np.zeros((args.ntry, 10001//100 +2))
    L_ess = np.zeros((args.ntry, 10001//100 +2))
        
    for k in range(args.ntry):
        model, L = swvi_nf(10001, V, 1e-3, plot=False, n_particles=500, dt_mcmc=1e-1)

        for i in range(len(L)):
            z = torch.randn((500, 3), device=device)
            z = F.normalize(z, p=2, dim=-1)

            x, log_det = L[i](z)
            log_prob = np.log(1 / (4 * np.pi)) * np.ones(z.shape[0]) - log_det.detach().cpu().numpy()

            _,  kl, ess = kl_ess(log_prob, s2_target(x[-1].detach().cpu().numpy()))

            L_kl[k, i] = kl
            L_ess[k, i] = ess/z.shape[0] * 100
            
            
        
    if args.sw_sphere:
        np.savetxt("./kl_sw_sphere", L_kl, delimiter=",")
        np.savetxt("./ess_sw_sphere", L_ess, delimiter=",")
    else:
        np.savetxt("./kl_sw", L_kl, delimiter=",")
        np.savetxt("./ess_sw", L_ess, delimiter=",")

        
        
        
