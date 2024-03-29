import torch

import numpy as np
import torch.nn.functional as F



def roll_by_gather(mat,dim, shifts: torch.LongTensor):
    ## https://stackoverflow.com/questions/66596699/how-to-shift-columns-or-rows-in-a-tensor-with-different-offsets-in-pytorch

    # assumes 2D array
    n_rows, n_cols = mat.shape
    
    if dim==0:
        arange1 = torch.arange(n_rows, device=mat.device).view((n_rows, 1)).repeat((1, n_cols))
        arange2 = (arange1 - shifts) % n_rows
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols, device=mat.device).view(( 1,n_cols)).repeat((n_rows,1))
        arange2 = (arange1 - shifts) % n_cols
        return torch.gather(mat, 1, arange2)
    

def dCost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()
    
    n = u_values.shape[-1]
    m_batch, m = v_values.shape
    
    v_cdf_theta = v_cdf -(theta - torch.floor(theta))
    
    mask_p = v_cdf_theta>=0
    mask_n = v_cdf_theta<0
         
    v_values[mask_n] += torch.floor(theta)[mask_n]+1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    ## ??
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1
    
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))

    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1,1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1,1))
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)
    
    u_index = torch.searchsorted(u_cdf, v_cdf_theta)
    u_icdf_theta = torch.gather(u_values, -1, u_index.clip(0, n-1))
    
    ## Deal with 1
    u_cdfm = torch.cat([u_cdf, u_cdf[:,0].view(-1,1)+1], dim=1)
    u_valuesm = torch.cat([u_values, u_values[:,0].view(-1,1)+1],dim=1)
    u_indexm = torch.searchsorted(u_cdfm, v_cdf_theta, right=True)
    u_icdfm_theta = torch.gather(u_valuesm, -1, u_indexm.clip(0, n))
    
    dCp = torch.sum(torch.pow(torch.abs(u_icdf_theta-v_values[:,1:]), p)
                   -torch.pow(torch.abs(u_icdf_theta-v_values[:,:-1]), p), axis=-1)
    
    dCm = torch.sum(torch.pow(torch.abs(u_icdfm_theta-v_values[:,1:]), p)
                   -torch.pow(torch.abs(u_icdfm_theta-v_values[:,:-1]), p), axis=-1)
    
    return dCp.reshape(-1,1), dCm.reshape(-1,1)


def Cost(theta, u_values, v_values, u_cdf, v_cdf, p):
    v_values = v_values.clone()
    
    m_batch, m = v_values.shape
    n_batch, n = u_values.shape

    v_cdf_theta = v_cdf -(theta - torch.floor(theta))
    
    mask_p = v_cdf_theta>=0
    mask_n = v_cdf_theta<0
    
    v_values[mask_n] += torch.floor(theta)[mask_n]+1
    v_values[mask_p] += torch.floor(theta)[mask_p]
    ## ??
    if torch.any(mask_n) and torch.any(mask_p):
        v_cdf_theta[mask_n] += 1
    
    v_cdf_theta2 = v_cdf_theta.clone()
    v_cdf_theta2[mask_n] = np.inf
    shift = (-torch.argmin(v_cdf_theta2, axis=-1))# .tolist()

    v_cdf_theta = roll_by_gather(v_cdf_theta, 1, shift.view(-1,1))
    v_values = roll_by_gather(v_values, 1, shift.view(-1,1))
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)  
        
    cdf_axis, cdf_axis_sorter = torch.sort(torch.cat((u_cdf, v_cdf_theta), -1), -1)
    cdf_axis_pad = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis_pad[..., 1:] - cdf_axis_pad[..., :-1]

    u_index = torch.searchsorted(u_cdf, cdf_axis)
    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
        
    v_values = torch.cat([v_values, v_values[:,0].view(-1,1)+1], dim=1)
    v_index = torch.searchsorted(v_cdf_theta, cdf_axis)
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m))
    
    if p == 1:
        ot_cost = torch.sum(delta*torch.abs(u_icdf-v_icdf), dim=-1)
    elif p == 2:
        ot_cost = torch.sum(delta*torch.square(u_icdf-v_icdf), dim=-1)
    else:
        ot_cost = torch.sum(delta*torch.pow(torch.abs(u_icdf-v_icdf), p), dim=-1)
    return ot_cost


def emd1D_circle(u_values, v_values, u_weights=None, v_weights=None, p=1, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    if p == 1:
        ## Code inspired from https://gitlab.gwdg.de/shundri/circularOT/-/tree/master/
        values_sorted, values_sorter = torch.sort(torch.cat((u_values, v_values), -1), -1)
        
        cdf_diff = torch.cumsum(torch.gather(torch.cat((u_weights, -v_weights),-1),-1,values_sorter),-1)
        cdf_diff_sorted, cdf_diff_sorter = torch.sort(cdf_diff, axis=-1)
        
        values_sorted = torch.nn.functional.pad(values_sorted, (0,1), value=1)
        delta = values_sorted[..., 1:]-values_sorted[..., :-1]
        weight_sorted = torch.gather(delta, -1, cdf_diff_sorter)

        sum_weights = torch.cumsum(weight_sorted, axis=-1)-0.5
        sum_weights[sum_weights<0] = np.inf
        inds = torch.argmin(sum_weights, axis=-1)
            
        levMed = torch.gather(cdf_diff_sorted, -1, inds.view(-1,1))
        
        return torch.sum(delta * torch.abs(cdf_diff - levMed), axis=-1)
    else:
        raise NotImplementedError(p)


def binary_search_circle(u_values, v_values, u_weights=None, v_weights=None, p=1, 
                         Lm=10, Lp=10, tm=-1, tp=1, eps=1e-6, require_sort=True):
    r"""
    Computes the Wasserstein distance on the circle using the Binary search algorithm proposed in [1].

    Parameters:
    u_values : ndarray, shape (n_batch, n_samples_u)
        samples in the source domain
    v_values : ndarray, shape (n_batch, n_samples_v)
        samples in the target domain
    u_weights : ndarray, shape (n_batch, n_samples_u), optional
        samples weights in the source domain
    v_weights : ndarray, shape (n_batch, n_samples_v), optional
        samples weights in the target domain
    p : float, optional
        Power p used for computing the Wasserstein distance
    Lm : int, optional
        Lower bound dC
    Lp : int, optional
        Upper bound dC
    tm: float, optional
        Lower bound theta
    tp: float, optional
        Upper bound theta
    eps: float, optional
        Stopping condition
    require_sort: bool, optional
        If True, sort the values.

    [1] Delon, Julie, Julien Salomon, and Andrei Sobolevski. "Fast transport optimization for Monge costs on the circle." SIAM Journal on Applied Mathematics 70.7 (2010): 2239-2258.
    """
    ## Matlab Code : https://users.mccme.ru/ansobol/otarie/software.html
    
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]
    
    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)
    
    L = max(Lm,Lp)
    
    tm = tm * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1,1)
    tm = tm.repeat(1, m)
    tp = tp * torch.ones((u_values.shape[0],), dtype=dtype, device=device).view(-1,1)
    tp = tp.repeat(1, m)
    tc = (tm+tp)/2
    
    done = torch.zeros((u_values.shape[0],m))
        
    cpt = 0
    while torch.any(1-done):
        cpt += 1
        
        dCp, dCm = dCost(tc, u_values, v_values, u_cdf, v_cdf, p)
        done = ((dCp*dCm)<=0) * 1
        
        mask = ((tp-tm)<eps/L) * (1-done)
        
        if torch.any(mask):
            ## can probably be improved by computing only relevant values
            dCptp, dCmtp = dCost(tp, u_values, v_values, u_cdf, v_cdf, p)
            dCptm, dCmtm = dCost(tm, u_values, v_values, u_cdf, v_cdf, p)
            Ctm = Cost(tm, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
            Ctp = Cost(tp, u_values, v_values, u_cdf, v_cdf, p).reshape(-1, 1)
        
            mask_end = mask * (torch.abs(dCptm-dCmtp)>0.001)
            tc[mask_end>0] = ((Ctp-Ctm+tm*dCptm-tp*dCmtp)/(dCptm-dCmtp))[mask_end>0]
            done[torch.prod(mask, dim=-1)>0] = 1
        ## if or elif?
        elif torch.any(1-done):
            tm[((1-mask)*(dCp<0))>0] = tc[((1-mask)*(dCp<0))>0]
            tp[((1-mask)*(dCp>=0))>0] = tc[((1-mask)*(dCp>=0))>0]
            tc[((1-mask)*(1-done))>0] = (tm[((1-mask)*(1-done))>0]+tp[((1-mask)*(1-done))>0])/2

    # print(f"cpt = {cpt}")
    
    return Cost(tc.detach(), u_values, v_values, u_cdf, v_cdf, p)


def sliced_cost(Xs, Xt, Us, p=2, u_weights=None, v_weights=None, method=0):
    """
        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        Xt: ndarray, shape (n_samples_v, dim)
            Samples in the target domain
        Us: ndarray, shape (num_projections, d, 2)
            Independent samples of the Uniform distribution on V_{d,2}
        p: float
            Power
        method: int
            0 for binary search, 1 for level median
    """
    n_projs, d, k = Us.shape
    n, _ = Xs.shape
    m, _ = Xt.shape
    
    ## To project on the great circle S^{d-1}\cap span(UU^T), but here, we only need the coordinates on S^1
    #P = torch.matmul(Us, torch.transpose(Us,1,2))
    #Projection on plane
    #Xps = torch.matmul(P[:,None], Xs[:,:,None]).reshape(n_projs, n, d)
    #Xpt = torch.matmul(P[:,None], Xt[:,:,None]).reshape(n_projs, m, d)
    #Projection on sphere
    #Xps = F.normalize(Xps, p=2, dim=-1)
    #Xpt = F.normalize(Xpt, p=2, dim=-1)
    
#     X0 = torch.tensor([1,0], dtype=torch.float).reshape(-1,1)
#     Xp0 = torch.matmul(Us, X0)
    
    # if torch.isnan(Xs).sum().item() > 0:
    #     raise Exception(f"Xps is NaN {Xs}")

    # if torch.isnan(Xt).sum().item() > 0:
    #     raise Exception(f"Xpt is NaN {Xt}")

    ## Projection on S^1
    ## Projection on plane
    Xps = torch.matmul(torch.transpose(Us,1,2)[:,None], Xs[:,:,None]).reshape(n_projs, n, 2)
    Xpt = torch.matmul(torch.transpose(Us,1,2)[:,None], Xt[:,:,None]).reshape(n_projs, m, 2)
    ## Projection on sphere
    Xps = F.normalize(Xps, p=2, dim=-1)
    Xpt = F.normalize(Xpt, p=2, dim=-1)
    
    ## Get coords
#     Xps = torch.arccos(torch.clamp(torch.sum(Xps[:,0][:,None]*Xps, dim=-1), min=-1, max=1))/(2*np.pi)
#     Xpt = torch.arccos(torch.clamp(torch.sum(Xpt[:,0][:,None]*Xpt, dim=-1), min=-1, max=1))/(2*np.pi)
    # eps = 1e-5
#     Xps = torch.arccos(torch.clamp(torch.sum(Xps[:,0][:,None]*Xps, dim=-1), min=-1+eps, max=1-eps))/(2*np.pi)
#     Xpt = torch.arccos(torch.clamp(torch.sum(Xpt[:,0][:,None]*Xpt, dim=-1), min=-1+eps, max=1-eps))/(2*np.pi)
    
    
#     Xtest = torch.arccos(torch.sum(Xps[:,0][:,None]*Xps, dim=-1))/(2*np.pi)
#     print(Xtest[0])

    # x0 = torch.tensor([1, 0], dtype=torch.float, device=Xps.device).reshape(1,-1)

    # scalar_product_xs = torch.sum(x0*Xps, dim=-1)
    # scalar_product_xt = torch.sum(x0*Xpt, dim=-1)

    # Xps = torch.arccos(torch.clamp(scalar_product_xs, min=-1+eps, max=1-eps))/(2*np.pi)
    # Xpt = torch.arccos(torch.clamp(scalar_product_xt, min=-1+eps, max=1-eps))/(2*np.pi)

    # if torch.isnan(Xps).sum().item() > 0:
    #     raise Exception(f"Xps is NaN {Xps}")

    # if torch.isnan(Xpt).sum().item() > 0:
    #     raise Exception(f"Xpt is NaN {Xpt}")


    Xps = (torch.pi + torch.atan2(-Xps[:,:,1], -Xps[:,:,0])) / (2 * torch.pi)
    Xpt = (torch.pi + torch.atan2(-Xpt[:,:,1], -Xpt[:,:,0])) / (2 * torch.pi)

#     print(Xps.shape, Xpt.shape)
#     print(Xps[0])
#     print(Xpt[0])

    # if torch.isnan(Xps).sum().item() > 0:
    #     raise Exception(f"Xps is NaN {Xps}")

    # if torch.isnan(Xpt).sum().item() > 0:
    #     raise Exception(f"Xpt is NaN {Xpt}")

    func = binary_search_circle if method == 0 else emd1D_circle 
    w1 = func(Xps, Xpt, p=p, u_weights=u_weights, v_weights=v_weights)

    return torch.mean(w1)


def sliced_wasserstein_sphere(Xs, Xt, num_projections, u_weights=None, v_weights=None, p=2, method=0):
    """
        Compute the sliced-Wasserstein distance on the sphere.

        Parameters:
        Xs: ndarray, shape (n_samples_u, dim)
            Samples in the source domain
        Xt: ndarray, shape (n_samples_v, dim)
            Samples in the target domain
        num_projections: int
            Number of projections
        device: str
        p: float
            Power of SW. Need to be >= 1.
        method: int
            0 for binary search, 1 for level median.
    """
    d = Xs.size(1)

    ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
    Z = torch.randn((num_projections,d,2), device=Xs.device)
    U, _ = torch.linalg.qr(Z)

    return sliced_cost(Xs, Xt, U, p=p, u_weights=u_weights, v_weights=v_weights, method=method)


def sliced_cost_unif(Xs, Us):
    n_projs, d, k = Us.shape
    n, _ = Xs.shape

    ## Projection on S^1
    ## Projection on plane
    Xps = torch.matmul(torch.transpose(Us,1,2)[:,None], Xs[:,:,None]).reshape(n_projs, n, 2)
    ## Projection on sphere
    Xps = F.normalize(Xps, p=2, dim=-1)

    Xps = (torch.pi + torch.atan2(-Xps[:,:,1], -Xps[:,:,0])) / (2 * torch.pi)

    w2 = w2_unif_circle(Xps)

    return torch.mean(w2)


def sliced_wasserstein_sphere_uniform(Xs, num_projections):
    """
    Compute the sliced-Wasserstein distance on the sphere with uniform distrib.

    """
    d = Xs.shape[1]

    ## Uniforms and independent samples on the Stiefel manifold V_{d,2}
    Z = torch.randn((num_projections,d,2), device=Xs.device)
    U, _ = torch.linalg.qr(Z)

    return sliced_cost_unif(Xs, U)


def w2_unif_circle(u_values):
    """
    	Closed-form

        weights 1/n
        Compute u_values vs Uniform distribution

        Parameters:
        u_values: ndarray, shape (n_batch, n_samples)
    """

    n = u_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    u_values, _ = torch.sort(u_values, -1)
    u_weights = torch.full((n,), 1/n, dtype=u_values.dtype, device=u_values.device)
    u_cdf = torch.cumsum(u_weights, -1)

    cpt1 = torch.mean(u_values**2, axis=-1)
    x_mean = torch.mean(u_values, axis=-1)

    ns_n2 = torch.arange(n-1, -n, -2, dtype=torch.float, device=u_values.device)/n**2
    cpt2 = torch.sum(ns_n2 * u_values, dim=-1)

    return cpt1 - x_mean**2 +cpt2 + 1/12
