import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .distributions import *
from .utils_sphere import *
from .utils_vmf import *


def val_mnist(model, device, latent_dim=2, latent_distr="unif"):
    model.eval()

    torch.manual_seed(42)
    r,c = 5,5
    
    if latent_distr == "unif_sphere": ## unif sphere
        z = torch.randn(r*c, latent_dim, device=device)
        z = F.normalize(z, p=2, dim=-1)
    elif latent_distr == "unif":
        z = -1+2*torch.rand(r*c, latent_dim, device=device)
    elif latent_distr == "ring":
        # g = torch.randn(n, d, device=device)
        # z = F.normalize(g, p=2, dim=1)
        z = rand_ring2d(r*c).to(device)
    elif latent_distr == "circle":
        z = rand_circle2d(r*c).to(device)
    elif latent_distr == "vmf":
        mu = torch.tensor([1,0,0], dtype=torch.float64)
        kappa = 10
        X = rand_von_mises_fisher(mu, kappa=kappa, N=r*c)
        z = torch.tensor(X, device=device, dtype=torch.float)
    elif latent_distr == "mixture_vmf":
        ps = np.ones(6)/6
        mus = torch.tensor([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,-1,0],[0,0,-1]], dtype=torch.float64)
        mus = F.normalize(mus, p=2, dim=-1)
        Z = np.random.multinomial(r*c,ps)
        X = []
#         y = []
        for k in range(len(Z)):
            if Z[k]>0:
                vmf = rand_von_mises_fisher(mus[k], kappa=10, N=int(Z[k]))
                X += list(vmf)
#             y += list(k*np.ones(len(vmf)))
        z = torch.tensor(X, device=device, dtype=torch.float)
        
    gen_imgs = model.decoder(z).reshape(-1,28,28).detach().cpu()

    cpt = 0
    fig,ax = plt.subplots(r,c)
    for i in range(r):
        for j in range(c):
            ax[i,j].imshow(gen_imgs[cpt],"gray")
            ax[i,j].axis('off')

            cpt += 1
                
    fig.set_size_inches(6, 6)
    plt.tight_layout()
    plt.show()


def plot_latent(model, test_loader, device):
    model.eval()

    test_encode, test_targets = [], []
    for x_val, y_val in test_loader:
        x_val = x_val.to(device)

        zhat = model.encoder(x_val)
        # yhat = model.decoder(zhat)
        test_encode.append(zhat.detach())
        test_targets.append(y_val.detach())
    
    test_encode = torch.cat(test_encode).cpu().numpy()
    test_targets = torch.cat(test_targets).cpu().numpy()
    
    # Distribution of the encoded samples
    z = test_encode
    Y = test_targets

    plt.figure(figsize=(10,10))
    plt.scatter(z[:,0], -z[:,1], c=10*Y, cmap=plt.cm.Spectral)
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.show()


def plot_latent_sphere(model, test_loader, device):
    model.eval()

    test_encode, test_targets = [], []
    for x_val, y_val in test_loader:
        x_val = x_val.to(device)

        zhat = model.encoder(x_val)
        # yhat = model.decoder(zhat)
        test_encode.append(zhat.detach())
        test_targets.append(y_val.detach())
            
    test_encode = torch.cat(test_encode).cpu().numpy()
    test_targets = torch.cat(test_targets).cpu().numpy()
    
    # Distribution of the encoded samples
    z = test_encode
    Y = test_targets

    plt.figure(figsize=(10,10))
    ax = plt.axes(projection='3d')
    plot_3d_scatter(z, ax, colour=10*Y)

#     plt.scatter(z[:,0], -z[:,1], c=10*Y, cmap=plt.cm.Spectral)
    plt.xlim([-1.5,1.5])
    plt.ylim([-1.5,1.5])
    plt.show()