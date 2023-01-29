import sys
import torch
import ot
import argparse

import torch.nn.functional as F
import torch.nn as nn
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


device = "cuda" if torch.cuda.is_available() else "cpu"


parser = argparse.ArgumentParser()
parser.add_argument("--loss", type=str, default="ssw", help="Which loss to use")
parser.add_argument("--dataset", type=str, default="fire", help="Which dataset to use")
parser.add_argument("--n_projs", type=int, default=1000, help="Number of projections")
parser.add_argument("--pbar", action="store_true", help="If yes, plot pbar")
parser.add_argument("--batch_size", type=int, default=15000, help="Batch size")
parser.add_argument("--n_epochs", type=int, default=5001, help="Number of epochs")
parser.add_argument("--lr", type=float, default=1e-1, help="Learning Rate")
parser.add_argument("--n_blocks", type=int, default=48, help="Number of blocks in the NF")
parser.add_argument("--n_components", type=int, default=100, help="Number of components in the NF")
parser.add_argument("--n_try", type=int, default=5, help="Number of iterations")
args = parser.parse_args()


config = {
    "training_size":0.7,
    "validation_size":0,
    "test_size":0.3,
    "name":args.dataset
}

eps = 1e-5


def loss_kl(h, log_det):
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


        model = create_NF(3, args.n_blocks, args.n_components).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        
        if args.batch_size>len(train_loader.dataset):
            batch_size = len(train_loader.dataset) # 500
        else:
            batch_size = args.batch_size

        if args.pbar:
            pbar = trange(n_steps)
        else:
            pbar = range(n_steps)

        test_nll = []

        for k in pbar:
            for data, _ in train_loader:
                optimizer.zero_grad()

                X_target = data.to(device)
                x, log_det = model(X_target)

                if args.loss == "ssw":
                    loss = sliced_wasserstein_sphere_unif(x[-1], num_projections, device)
                elif args.loss == "ssw_mbot":
                    loss = sliced_wasserstein_sphere_unif(x[-1], num_projections, device)
                elif args.loss == "sw":
                    z = F.normalize(torch.randn(batch_size, 3, device=device), p=2, dim=-1)
                    loss = sliced_wasserstein(x[-1], z, num_projections, device, p=2)
                elif args.loss == "mbot":
                    z = F.normalize(torch.randn(batch_size, 3, device=device), p=2, dim=-1)
                    ip = x[-1]@z.T
                    M = torch.arccos(torch.clamp(ip, min=-1+1e-5, max=1-1e-5))
                    a = torch.ones(x[-1].shape[0], device=device) / x[-1].shape[0]
                    b = torch.ones(z.shape[0], device=device) / z.shape[0]
                    loss = ot.emd2(a, b, M)
                elif args.loss == "kl":
                    loss = loss_kl(x[-1], log_det)
                elif args.loss == "kl+ssw":
                    ssw = sliced_wasserstein_sphere_unif(x[-1], num_projections, device)
                    kl = loss_kl(x[-1], log_det)
                    loss = kl + 10 * ssw
                
                
                loss.backward()
                optimizer.step()


                if args.pbar:
                    pbar.set_postfix_str(f"loss = {loss.item():.3f}")

                z, log_det = model(test_data.to(device))
                density = log_likelihood(z[-1], log_det).detach().cpu()
                test_nll.append(-density.mean())


        #     scheduler.step(loss)
        L_density[i] = -density.mean()
        print(k, L_density[i], flush=True)
    
        torch.save(model.state_dict(), "./weights/nf_density_"+args.dataset+"_"+args.loss+"_"+str(i)+".model")
        np.savetxt("./Results/evol_nll_"+args.dataset+"_"+args.loss+"_"+str(i), test_nll)
        
    print("Mean", np.mean(L_density), np.std(L_density), flush=True)
    np.savetxt("./Results/NLL_"+args.dataset+"_"+args.loss, L_density)

              
