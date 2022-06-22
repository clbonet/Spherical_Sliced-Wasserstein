import torch
import ot

import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm.auto import trange

from .sw_sphere import sliced_wasserstein_sphere
from .sw import sliced_wasserstein
from .utils_plot import *
from .utils_vmf import rand_von_mises_fisher
from .distributions import *

criterion = nn.BCELoss(reduction='mean')


def train(model, optimizer, n_epochs, train_loader, test_loader, 
            ae_loss, device, latent_distr="unif", plot_val=False,
            plot_results=False, bar=False):
    
    print(latent_distr, flush=True)
    if bar:
        pbar = trange(n_epochs)
    else:
        pbar = range(n_epochs)

    losses = []
    val_losses = []
    w_latent_losses = []
    w_losses = []
    sw_latent_losses = []
    sw_sphere_latent_losses = []

    for e in pbar:
        loss_epoch = 0
        cpt_batch = 0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)

            model.train()

            z_hat = model.encoder(x_batch)            
            y_hat = model.decoder(z_hat)

            l = ae_loss(x_batch, y_hat, z_hat, latent_distr)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_epoch += l.item()/x_batch.size(0)
            cpt_batch += 1

        losses.append(loss_epoch/cpt_batch)


        loss_val_epoch = 0
        cpt_batch = 0
        cpt_w, cpt_w_latent, cpt_sw_latent, cpt_sw_sphere_latent = 0, 0, 0, 0

        with torch.no_grad():
            for x_val, _ in test_loader:
                x_val = x_val.to(device)

                model.eval()
                zhat = model.encoder(x_val)
                
                if latent_distr == "unif_sphere" or latent_distr == "mixture_vmf" or latent_distr == "vmf":
                    z_hat = F.normalize(z_hat, p=2, dim=-1)
                
                yhat = model.decoder(zhat)
                val_l = ae_loss(x_val, yhat, zhat, latent_distr)
                loss_val_epoch += val_l.item()/x_val.size(0)
                cpt_batch += 1
                
                n, d = zhat.shape
                
                if latent_distr == "unif_sphere":
                    target_latent = torch.randn(zhat.shape[0], zhat.shape[1], device=device)
                    target_latent = F.normalize(target_latent, p=2, dim=-1)
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
                elif latent_distr == "vmf":
                    mu = torch.tensor([1,0,0], dtype=torch.float64)
                    kappa = 10
                    X = rand_von_mises_fisher(mu, kappa=kappa, N=n)
                    target_latent = torch.tensor(X, device=device, dtype=torch.float)
                elif latent_distr == "unif":
                    target_latent = -1+2*torch.rand(n, d, device=device)
                elif latent_distr == "ring":
                    target_latent = rand_ring2d(n).to(device)
                elif latent_distr == "circle":
                    target_latent = rand_circle2d(n).to(device)
                
                if latent_distr == "unif_sphere" or latent_distr == "mixture_vmf" or latent_distr == "vmf":
                    ip = zhat@target_latent.T
                    M = torch.arccos(torch.clamp(ip, min=-1+1e-5, max=1-1e-5))
                else:
                    M = ot.dist(zhat, target_latent, metric="sqeuclidean")
                    
                a = torch.ones(zhat.shape[0]) / zhat.shape[0]
                b = torch.ones(target_latent.shape[0]) / target_latent.shape[0]
                cpt_w_latent += ot.emd2(a, b, M).item()
                
                M = ot.dist(x_val.reshape(n, -1), yhat.reshape(n, -1), metric="sqeuclidean")
                a = torch.ones(x_val.shape[0]) / x_val.shape[0]
                b = torch.ones(yhat.shape[0]) / yhat.shape[0]
                cpt_w += ot.emd2(a, b, M).item()

                if latent_distr == "unif_sphere" or latent_distr == "mixture_vmf" or latent_distr == "vmf":
                    cpt_sw_sphere_latent += sliced_wasserstein_sphere(zhat, target_latent, 1000, device, p=2).item()
                cpt_sw_latent += sliced_wasserstein(zhat, target_latent, 1000, device, p=2).item()


            val_losses.append(loss_val_epoch/cpt_batch)
            w_latent_losses.append(cpt_w_latent/cpt_batch)
            w_losses.append(cpt_w/cpt_batch)
            sw_latent_losses.append(cpt_sw_latent/cpt_batch)
            sw_sphere_latent_losses.append(cpt_sw_sphere_latent/cpt_batch)
        
        if bar:
            pbar.set_postfix_str(f"loss = {losses[-1]:.3f}, val_loss = {val_losses[-1]:.3f}, w_loss = {w_losses[-1]:.3f}, sw_latent_loss={sw_latent_losses[-1]:.3f}")

        if e%10 == 0 and plot_results:  
            with torch.no_grad():
                model.eval()
                for x_val, _ in test_loader:
                    fig, ax = plt.subplots(1,2,figsize=(10,10))

                    ax[0].imshow(x_val[0][0],"gray")

                    x_val = x_val.to(device)

                    model.eval()
                    zhat = model.encoder(x_val[0][0].reshape(-1,28,28))
                    zhat = F.normalize(zhat, p=2, dim=-1)
                    yhat = model.decoder(zhat).reshape(-1,1,28,28)
                    ax[1].imshow(yhat[0][0].cpu().detach().numpy(),"gray")
                    plt.show()

                    break
                
                if latent_distr == "unif_sphere" or latent_distr == "mixture_vmf" or latent_distr == "vmf":
                    plot_latent_sphere(model, test_loader, device)
                else:
                    plot_latent(model, test_loader, device)

    if plot_val:
        plt.plot(losses, label="Training loss")
        plt.plot(val_losses, label="Validation loss")
        plt.legend()
        plt.show()
        
    return w_latent_losses, w_losses, sw_latent_losses, sw_sphere_latent_losses

