import sys
import torch
import argparse
import time
import ot

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("../lib")

from sw_sphere import sliced_wasserstein_sphere, sliced_wasserstein_sphere_unif
from utils_sphere import *
from utils_vmf import *


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=20, help="number of restart")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"




if __name__ == "__main__":    
    kappa = 10
    ds = [3] #, 100, 500, 1000]
    # samples = range(1000, 10001, 1000) #[1000,2000,3000,5000,10000,2000,50000]
    samples = [int(1e2),int(1e3),int(1e4),int(1e5/2),int(1e5)] #,int(1e6/2)]
    projs = [200] #[50, 200] #, 500]

    L_ssw2 = np.zeros((len(ds), len(projs), len(samples), args.ntry))
    L_unif = np.zeros((len(ds), len(projs), len(samples), args.ntry))
    L_ssw1 = np.zeros((len(ds), len(projs), len(samples), args.ntry))

    L_w = np.zeros((len(ds), len(samples), args.ntry))
    L_s = np.zeros((len(ds), len(samples), args.ntry))

    for i, d in enumerate(ds):
        print(d, flush=True)
        mu = np.ones((d,))
        mu = mu/np.linalg.norm(mu)
        
        for k, n_samples in enumerate(samples):
            print(n_samples, flush=True)
            x0 = torch.randn((n_samples, d), device=device)
            x0 = F.normalize(x0, p=2, dim=-1)
        
            x1 = rand_von_mises_fisher(mu, kappa=kappa, N=n_samples)

            # print(x0.shape, x1.shape)
        
            for j in range(args.ntry):
                for l, n_projs in enumerate(projs):
                    try:
                        t0 = time.time()
                        sw = sliced_wasserstein_sphere(x0, torch.tensor(x1, dtype=torch.float, device=device), n_projs, device, p=2)
                        L_ssw2[i,l,k,j] = time.time()-t0
                    except:
                        L_ssw2[i,l,k,j] = np.inf
                        
                    try:
                        t0 = time.time()
                        sw = sliced_wasserstein_sphere(x0, torch.tensor(x1, dtype=torch.float, device=device), n_projs, device, p=1)
                        L_ssw1[i,l,k,j] = time.time()-t0
                    except:
                        L_ssw1[i,l,k,j] = np.inf
                        
                    try:
                        t0 = time.time()
                        sw = sliced_wasserstein_sphere_unif(torch.tensor(x1, dtype=torch.float, device=device), n_projs, device)
                        L_unif[i,l,k,j] = time.time()-t0
                    except:
                        L_unif[i,l,k,j] = np.inf

                try:
                    t2 = time.time()
                    ip = x0@torch.tensor(x1, dtype=torch.float, device=device).T
                    M = torch.arccos(torch.clamp(ip, min=-1+1e-5, max=1-1e-5))
                    a = torch.ones(x0.shape[0], device=device) / x0.shape[0]
                    b = torch.ones(x1.shape[0], device=device) / x1.shape[0]
                    w = ot.sinkhorn2(a, b, M, reg=1, numitermax=10000, stopThr=1e-15).item()
                    L_s[i,k,j] = time.time()-t2
                except:
                    L_s[i,k,j] = np.inf

                try:
                    t1 = time.time()
                    ip = x0@torch.tensor(x1, dtype=torch.float, device=device).T
                    M = torch.arccos(torch.clamp(ip, min=-1+1e-5, max=1-1e-5))
                    a = torch.ones(x0.shape[0], device=device) / x0.shape[0]
                    b = torch.ones(x1.shape[0], device=device) / x1.shape[0]
                    w = ot.emd2(a, b, M).item()
                    L_w[i,k,j] = time.time()-t1
                except:
                    L_w[i,k,j] = np.inf


    for l, n_projs in enumerate(projs):
        np.savetxt("./Comparison_SSW_projs_"+str(n_projs), L_ssw2[0, l])
        np.savetxt("./Comparison_SSW1_projs_"+str(n_projs), L_ssw1[0, l])
        np.savetxt("./Comparison_SSW2_unif_projs_"+str(n_projs), L[0, l])
    np.savetxt("./Comparison_SW_W", L_w[0])
    np.savetxt("./Comparison_SW_Sinkhorn", L_s[0])
        
        
