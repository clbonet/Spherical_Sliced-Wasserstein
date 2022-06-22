import sys
import torch
import torchvision
import argparse
import time

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from geomloss import SamplesLoss 

sys.path.append("../lib")

from swae.nn import AE
from swae.train_swae import train
from swae.distributions import *
from swae.sw import sliced_wasserstein
from swae.fid_score import *

from sw_sphere import sliced_wasserstein_sphere, sliced_wasserstein_sphere_unif
from utils_vmf import rand_von_mises_fisher


parser = argparse.ArgumentParser()
parser.add_argument("--ntry", type=int, default=15, help="number of restart")
parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
parser.add_argument("--swae_sphere", help="If true, use sw_sphere", action="store_true")
parser.add_argument("--wae_mmd_imq", help="If true, use WAE-MMD + IMQ kernel", action="store_true")
parser.add_argument("--wae_mmd_rbf", help="If true, use WAE-MMD + RBF kernel", action="store_true")
parser.add_argument("--sae", help="If true, use SAE", action="store_true")
parser.add_argument("--gswae_circular", help="If True, use GSWAE with g circular", action="store_true")
parser.add_argument("--gswae_poly", help="If True, use GSWAE with g polynomial", action="store_true")
parser.add_argument("--prior", type=str, default="unif_sphere", help="Specify prior")
parser.add_argument("--d_latent", type=int, default=3, help="Dimension of the latent space")
parser.add_argument("--p", type=int, default=2, help="Order of SSW")
args = parser.parse_args()


transform = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()#,
                # torchvision.transforms.Normalize((0.5,), (0.5,))
    ])

train_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=True, transform=transform, download=True
)

test_dataset = torchvision.datasets.MNIST(
    root="~/torch_datasets", train=False, transform=transform, download=True
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=500, shuffle=True, num_workers=4, pin_memory=True
)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=4
)

real_cpu = torch.zeros((10000,28,28))

cpt = 0
for data, _ in test_loader:
    real_cpu[cpt:cpt+32] = data[:,0]
    cpt += 32
    
def compute_FID(model):
    n = 10000
    d = args.d_latent
    latent_distr = args.prior
    
    L = []
    for k in range(1):
        if latent_distr == "unif":
            z = -1+2*torch.rand(n, d, device=device)
        elif latent_distr == "ring":
            z = rand_ring2d(n).to(device)
        elif latent_distr == "circle":
            z = rand_circle2d(n).to(device)
        elif latent_distr == "unif_sphere": ## unif sphere
            target_latent = torch.randn(n, d, device=device)
            z = F.normalize(target_latent, p=2, dim=-1)
        elif latent_distr == "vmf":
            mu = torch.tensor([1,0,0], dtype=torch.float64)
            kappa = 10
            X = rand_von_mises_fisher(mu, kappa=kappa, N=n)
            z = torch.tensor(X, device=device, dtype=torch.float)
        elif latent_distr == "mixture_vmf":
            ps = np.ones(6)/6
            mus = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=torch.float64)
            mus = F.normalize(mus, p=2, dim=-1)
            Z = np.random.multinomial(n,ps)
            X = []
            for k in range(len(Z)):
                if Z[k]>0:
                    vmf = rand_von_mises_fisher(mus[k], kappa=10, N=int(Z[k]))
                    X += list(vmf)
            z = torch.tensor(X, device=device, dtype=torch.float)

        gen_imgs = torch.zeros((10000,28,28))
        for k in range(10):
            gen_imgs[1000*k:1000*(k+1)] = model.decoder(z[1000*k:1000*(k+1)]).detach().cpu().reshape(-1,28,28)

        t = time.time()
        fid = evaluate_fid_score(real_cpu.reshape(-1,28,28,1), gen_imgs.reshape(-1,28,28,1), batch_size=50)
        print("t=", time.time()-t)
        L.append(fid)
    return L



criterion = nn.BCELoss(reduction='mean')


def gaussian_kernel(x, y, h):
    return torch.exp(-torch.cdist(x,y)**2/h)

def imq(x, y, h):
    return h/(h+torch.norm(x.unsqueeze(1) - y.unsqueeze(0), dim=-1) ** 2)

def mmd(x, y, kernel, h):
    ## Unbiased estimate
    Kxx = kernel(x, x, h)
    Kyy = kernel(y, y, h)
    Kxy = kernel(x, y, h)

    n = x.shape[0]
    cpt1 = (torch.sum(Kxx)-torch.sum(Kxx.diag()))/(n-1) ## remove diag terms
    cpt2 = (torch.sum(Kyy)-torch.sum(Kyy.diag()))/(n-1)
    cpt3 = torch.sum(Kxy)/n

    return (cpt1+cpt2-2*cpt3)/n


def ae_loss(x, y, z, latent_distr="unif_sphere"):
    n, d = z.size()

    if latent_distr == "unif":
        target_latent = -1+2*torch.rand(n, d, device=device)
    elif latent_distr == "ring":
        target_latent = rand_ring2d(n).to(device)
    elif latent_distr == "circle":
        target_latent = rand_circle2d(n).to(device)
    elif latent_distr == "unif_sphere": ## unif sphere
        target_latent = torch.randn(n, d, device=device)
        target_latent = F.normalize(target_latent, p=2, dim=-1)
    elif latent_distr == "vmf":
        mu = torch.tensor([1,0,0], dtype=torch.float64)
        kappa = 10
        X = rand_von_mises_fisher(mu, kappa=kappa, N=n)
        target_latent = torch.tensor(X, device=device, dtype=torch.float)
    elif latent_distr == "mixture_vmf":
        ps = np.ones(6)/6
        mus = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=torch.float64)
        mus = F.normalize(mus, p=2, dim=-1)
        Z = np.random.multinomial(n,ps)
        X = []
        for k in range(len(Z)):
            if Z[k]>0:
                vmf = rand_von_mises_fisher(mus[k], kappa=10, N=int(Z[k]))
                X += list(vmf)
        target_latent = torch.tensor(X, device=device, dtype=torch.float)
    
    if args.swae_sphere and latent_distr=="unif_sphere":
        loss_latent = sliced_wasserstein_sphere_unif(z, 1000, device)
    elif args.swae_sphere:
        loss_latent = sliced_wasserstein_sphere(z, target_latent, 1000, device, p=args.p)
    elif args.wae_mmd_imq:
        h = 2 * d
        loss_latent = mmd(z, target_latent, imq, h)
    elif args.wae_mmd_rbf:
        h = 2 * d
        loss_latent = mmd(z, target_latent, gaussian_kernel, h)
    elif args.sae:
        loss_func = SamplesLoss("sinkhorn", blur=0.05,scaling = 0.95,diameter=0.01,debias=True)
        loss_latent = loss_func(z, target_latent)
    elif args.gswae_circular:
        loss_latent = sliced_wasserstein(z, target_latent, 1000, device, p=args.p, ftype="circular")
    elif args.gswae_poly:
        loss_latent = sliced_wasserstein(z, target_latent, 1000, device, p=args.p, ftype="poly")
    else:
        loss_latent = sliced_wasserstein(z, target_latent, 1000, device, p=args.p)
        
    reconstruction_loss = criterion(y, x)    
    return reconstruction_loss + 10.0*loss_latent


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print("device =", device,flush=True)
    
    d = args.d_latent
    print("d =", d, flush=True)

    L_w_losses = []
    L_sw_latent = []
    L_sw_sphere_latent = []
    L_w_latent = []
    L_fid = []

    for k in range(args.ntry):
        print("Try ",k)
        model = AE(16, d, normalize_output=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        w_latent_losses, w_losses, sw_latent_losses, sw_sphere_latent_losses = train(model, optimizer, args.epochs, 
                                                                                    train_loader, test_loader, ae_loss, 
                                                                                    device, args.prior, plot_results=False, 
                                                                                    bar=True)
        
        if args.swae_sphere:
            torch.save(model.state_dict(), "./results/swae_sphere_unif_"+args.prior+"_d_"+str(d)+"_"+str(k)+".model")
        elif args.sae:
            torch.save(model.state_dict(), "./results/sae_"+args.prior+"_d_"+str(d)+"_"+str(k)+".model")
        elif args.wae_mmd_imq:
            torch.save(model.state_dict(), "./results/wae_mmd_imq_"+args.prior+"_d_"+str(d)+"_"+str(k)+".model")
        elif args.wae_mmd_rbf:
            torch.save(model.state_dict(), "./results/wae_mmd_rbf_"+args.prior+"_d_"+str(d)+"_"+str(k)+".model")
        elif args.gswae_circular:
            torch.save(model.state_dict(), "./results/gswae_circular_"+args.prior+"_d_"+str(d)+"_"+str(k)+".model")
        elif args.gswae_poly:
            torch.save(model.state_dict(), "./results/gswae_poly_"+args.prior+"_d_"+str(d)+"_"+str(k)+".model")
        else:
            torch.save(model.state_dict(), "./results/swae_"+args.prior+"_d_"+str(d)+"_"+str(k)+".model")

        

        L_w_losses.append(w_losses)
        L_sw_latent.append(sw_latent_losses)
        L_sw_sphere_latent.append(sw_sphere_latent_losses)
        L_w_latent.append(w_latent_losses)
        
#        print("?")
#        L = compute_FID(model)
#        L_fid += L
        
#        print(k, "Fid="+str(np.mean(L))+"+/-"+str(np.std(L)), flush=True)

    L_w_losses = np.array(L_w_losses)
    L_sw_latent = np.array(L_sw_latent)
    L_sw_sphere_latent = np.array(L_sw_sphere_latent)
    L_w_latent = np.array(L_w_latent)


    if args.swae_sphere:
        np.savetxt("./results/Loss_w_sphere_"+args.prior+"_d_"+str(d), L_w_losses, delimiter=",")
        np.savetxt("./results/Loss_w_latent_sphere_"+args.prior+"_d_"+str(d), L_w_latent, delimiter=",")
        np.savetxt("./results/Loss_sws_sphere_"+args.prior+"_d_"+str(d), L_sw_sphere_latent, delimiter=",")
        np.savetxt("./results/Loss_sw_sphere_"+args.prior+"_d_"+str(d), L_sw_latent, delimiter=",")
        np.savetxt("swae_sphere_fid_"+args.prior+"_d_"+str(d), L_fid, delimiter=",")
        torch.save(model.state_dict(), "./results/swae_sphere_"+args.prior+"_d_"+str(d)+".model")
        
    elif args.wae_mmd_imq:
        np.savetxt("./results/Loss_w_wae_mmd_imq_"+args.prior+"_d_"+str(d), L_w_losses, delimiter=",")
        np.savetxt("./results/Loss_w_latent_wae_mmd_imq_"+args.prior+"_d_"+str(d), L_w_latent, delimiter=",")
        np.savetxt("./results/Loss_sws_wae_mmd_imq_"+args.prior+"_d_"+str(d), L_sw_sphere_latent, delimiter=",")
        np.savetxt("./results/Loss_sw_wae_mmd_imq_"+args.prior+"_d_"+str(d), L_sw_latent, delimiter=",")
        np.savetxt("wae_mmd_imq_sphere_fid_"+args.prior+"_d_"+str(d), L_fid, delimiter=",")
        torch.save(model.state_dict(), "./results/wae_mmd_imq_"+args.prior+"_d_"+str(d)+".model")
        
    elif args.wae_mmd_rbf:
        torch.save(model.state_dict(), "./results/wae_mmd_rbf_"+args.prior+"_d_"+str(d)+".model")
        
    elif args.sae:
        torch.save(model.state_dict(), "./results/sae"+args.prior+"_d_"+str(d)+".model")
        
    elif args.gswae_circular:
        torch.save(model.state_dict(), "./results/gswae_circular"+args.prior+"_d_"+str(d)+".model")
        
    elif args.gswae_poly:
        torch.save(model.state_dict(), "./results/gswae_poly"+args.prior+"_d_"+str(d)+".model")
        
    else:
        np.savetxt("./results/Loss_w_swae_"+args.prior+"_d_"+str(d), L_w_losses, delimiter=",")
        np.savetxt("./results/Loss_w_latent_swae_"+args.prior+"_d_"+str(d), L_w_latent, delimiter=",")
        np.savetxt("./results/Loss_sw_swae_"+args.prior+"_d_"+str(d), L_sw_latent, delimiter=",")
        np.savetxt("./results/Loss_sws_swae_"+args.prior+"_d_"+str(d), L_sw_sphere_latent, delimiter=",")
        np.savetxt("swae_fid"+args.prior+"_d_"+str(d), L_fid, delimiter=",")
        torch.save(model.state_dict(), "./results/swae_"+args.prior+"_d_"+str(d)+".model")
