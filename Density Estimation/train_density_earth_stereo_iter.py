import sys
import torch
import argparse

import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as D
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

from tqdm.auto import trange
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable

from datasets import EarthDataHandler, xyz_to_latlon

sys.path.append("../lib")
from utils_sphere import *
from exp_map_nf import create_NF
from sw_sphere import sliced_wasserstein_sphere, sliced_wasserstein_sphere_unif
from sw import sliced_wasserstein
from utils_vmf import rand_von_mises_fisher
from realnvp import create_RealNVP


device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="fire", help="Which dataset to use")
parser.add_argument("--n_projs", type=int, default=1000, help="Number of projections")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
parser.add_argument("--batch_size", type=int, default=15000, help="Batch size")
parser.add_argument("--n_epochs", type=int, default=10001, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
parser.add_argument("--n_try", type=int, default=5, help="Number of iterations")
parser.add_argument("--nh", type=int, default=25, help="Number of hidden units")
parser.add_argument("--nl", type=int, default=10, help="Number of layers")
args = parser.parse_args()


def stereographic_proj(x):
    return x[...,1:]/(1+x[...,0][:,None])

def inverse_stereographic_proj(x):
    norm_x2 = torch.linalg.norm(x, dim=-1, keepdim=True)**2
    return torch.cat([2*x/(norm_x2+1), 1-2/(norm_x2+1)], dim=-1)

base_distr = D.MultivariateNormal(torch.zeros(2,device=device),torch.eye(2,device=device))

def log_likelihood_stereo(z, log_det_f, y):
    d = y.shape[-1]
    
    prior = base_distr.log_prob(z)
    norm_y = torch.linalg.norm(y, dim=-1)
    log_det_stereo = d * torch.log(2/(norm_y**2+1))
    return prior + log_det - log_det_stereo

config = {
    "training_size":0.7,
    "validation_size":0,
    "test_size":0.3,
    "name":args.dataset
}

eps = 1e-5


def loss(h, log_det):
    prior = np.log(1 / (4 * np.pi)) * torch.ones(h.shape[0], device=device)
    return -(prior+log_det).mean()

def log_likelihood(h, log_det):
    prior = np.log(1 / (4 * np.pi)) * torch.ones(h.shape[0], device=device)
    return (prior+log_det)


if __name__ == "__main__":
    n_steps = args.n_epochs
    num_projections = args.n_projs
    
    L_density = np.zeros((args.n_try))
    
    for i in range(args.n_try):
    
        handler = EarthDataHandler(config, eps)
        # train_loader, val_loader, est_loader = handler.get_dataloaders(500, 500)
        train_loader, val_loader, est_loader = handler.get_dataloaders(args.batch_size, args.batch_size)

        for test_data, _ in val_loader:
            break


        model = create_RealNVP(nh=args.nh, nl=args.nl).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        batch_size = len(train_loader.dataset) # 500

        if args.pbar:
            pbar = trange(n_steps)
        else:
            pbar = range(n_steps)

        test_nll = []

        for k in pbar:
            for data, _ in train_loader:
                optimizer.zero_grad()

#                 X_target = data.to(device)
                X_target = stereographic_proj(data).to(device)
                x, _ = model(X_target)
        
                z = base_distr.sample((batch_size,))
                loss = sliced_wasserstein(x[-1], z, num_projections, device, p=2)

                loss.backward()
                optimizer.step()

                y = stereographic_proj(test_data).to(device)
                z, log_det = model(y)
                density = log_likelihood_stereo(z[-1], log_det, y).detach().cpu()
                test_nll.append(-density.mean())

                if args.pbar:
                    pbar.set_postfix_str(f"loss = {loss.item():.3f}" + f", nll = {test_nll[-1].item():.3f}")
                    
        L_density[i] = -density.mean()
        print(k, L_density[i], flush=True)
    
        torch.save(model.state_dict(), "./weights/nf_density_stereo_"+args.dataset+"_stereo_"+str(i)+".model")
        np.savetxt("./Results/evol_nll_stereo_"+args.dataset+"_stereo_"+str(i), test_nll)
        
    print("Mean", np.mean(L_density), np.std(L_density), flush=True)
    np.savetxt("./Results/NLL_stereo_"+args.dataset+"_stereo", L_density)

              
