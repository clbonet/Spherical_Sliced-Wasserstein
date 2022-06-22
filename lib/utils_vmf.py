import torch

import numpy as np
# import matplotlib.pyplot as plt
import numpy.matlib

from scipy.linalg import null_space
from scipy.special import iv, gamma


def pdf_vmf(x, mu, kappa):
    return torch.exp(kappa * torch.matmul(mu, x.T))[0]

## https://github.com/dlwhittenbury/von-Mises-Sampling/blob/master/von-Mises.ipynb
## https://github.com/dlwhittenbury/von-Mises-Fisher-Sampling/blob/master/von-Mises-Fisher.ipynb


def pdf_von_Mises(theta,mu,kappa):
    pdf = np.exp(kappa * np.cos(theta - mu)) / (2.0*np.pi*iv(0,kappa))
    return pdf


def rand_von_Mises(N,mu,kappa):

    """
        rand_von_Mises(N,mu,kappa)
        ==========================
        
        Generates theta an Nx1 array of samples of a von Mises distribution
        with mean direction mu and concentration kappa.

        INPUT:
         
            * N - number of samples to be generated (integer)
            * mu - mean direction (float)
            * kappa - concentration (float)

        OUTPUT:
         
            * theta - an Nx1 array of samples of a von Mises distribution with mean 
            direction mu and concentration kappa.

         References:
         ===========

         Algorithm first given in

         [1] D. J. Best and N. I. Fisher, Efficient Simulation of the von Mises
         Distribution, Applied Statistics, 28, 2, 152--157, (1979).

         Also given in the following textbook/monograph

         [2] N. I. Fisher, Statistical analysis of circular data, Cambridge University Press, (1993).

    """
    
    # Checks 
    # =======

    # N should be a positive non-zero integer scalar 
    if (type(N) is not int): 
        raise TypeError("N must be a positive non-zero integer.")
    if N <= 0:
        raise Exception("N must be a positive non-zero integer.")
        
    #  mu should be a real scalar. It can wrap around the circle, so it can be negative, positive and also 
    #  outside the range [0,2*pi].
    if (type(mu) is not float) and (type(mu) is not int):
        raise TypeError("mu must be a real scalar number.")

    # kappa should be positive real scalar 
    if (type(kappa) is not float) and (type(kappa) is not int): 
        raise TypeError("kappa must be a positive float.")
    if kappa < 0:
        raise Exception("kappa must be a positive float.")

    #  SPECIAL CASE
    # ==============

    #  As kappa -> 0 one obtains the uniform distribution on the circle
    float_epsilon = np.finfo(float).eps
    if kappa <= float_epsilon:
        theta = 2.0 * np.pi * np.random.rand(N,1) # [0,1] -> [0,2*pi]
        return theta

    # MAIN BODY OF ALGORITHM
    # =======================
    
    # Used same notation as Ref.~[2], p49
    
    a = 1.0 + np.sqrt(1.0 + 4.0 * kappa**2)
    b = (a - np.sqrt(2.0 * a)) / (2.0 * kappa)
    r = (1.0 + b**2) / (2.0 * b)

    counter = 0 
    theta = np.zeros((N,1)) 

    while counter <= N-1:
        
        # Pseudo-random numbers sampled from a uniform distribution [0,1]
        U1 = np.random.rand()
        U2 = np.random.rand()
        U3 = np.random.rand() 
        
        z = np.cos(np.pi * U1)
        f = (1.0 + r *z) / (r + z)
        c = kappa * (r - f)
        
        if ( ((c * (2.0 - c) - U2) > 0.0)  or ((np.log(c/U2) + 1.0 - c) > 0.0) ):
        
            theta[counter] = np.mod(np.sign(U3 - 0.5) * np.arccos(f) + mu, 2*np.pi)     
            counter += 1

    return theta


def rand_uniform_hypersphere(N,p):
    
    """ 
        rand_uniform_hypersphere(N,p)
        =============================
    
        Generate random samples from the uniform distribution on the (p-1)-dimensional 
        hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$. We use the method by 
        Muller [1], see also Ref. [2] for other methods.
        
        INPUT:  
        
            * N (int) - Number of samples 
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the 
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
    
    References:
    
    [1] Muller, M. E. "A Note on a Method for Generating Points Uniformly on N-Dimensional Spheres."
    Comm. Assoc. Comput. Mach. 2, 19-20, Apr. 1959.
    
    [2] https://mathworld.wolfram.com/SpherePointPicking.html
    
    """
    
    if (p<=0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")
    
    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")
    
    v = np.random.normal(0,1,(N,p))
    
#    for i in range(N):
#        v[i,:] = v[i,:]/np.linalg.norm(v[i,:])
        
    v = np.divide(v,np.linalg.norm(v,axis=1,keepdims=True))
    
    return v



def rand_t_marginal(kappa,p,N=1):
    """
        rand_t_marginal(kappa,p,N=1)
        ============================
        
        Samples the marginal distribution of t using rejection sampling of Wood [3]. 
    
        INPUT: 
        
            * kappa (float) - concentration        
            * p (int) - The dimension of the generated samples on the (p-1)-dimensional hypersphere.
                - p = 2 for the unit circle $\mathbb{S}^{1}$
                - p = 3 for the unit sphere $\mathbb{S}^{2}$
            Note that the (p-1)-dimensional hypersphere $\mathbb{S}^{p-1} \subset \mathbb{R}^{p}$ and the 
            samples are unit vectors in $\mathbb{R}^{p}$ that lie on the sphere $\mathbb{S}^{p-1}$.
            * N (int) - number of samples 
        
        OUTPUT: 
        
            * samples (array of floats of shape (N,1)) - samples of the marginal distribution of t
    """
    
    # Check kappa >= 0 is numeric 
    if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
        raise Exception("kappa must be a non-negative number.")
        
    if (p<=0) or (type(p) is not int):
        raise Exception("p must be a positive integer.")
    
    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")
    
    
    # Start of algorithm 
    b = (p - 1.0) / (2.0 * kappa + np.sqrt(4.0 * kappa**2 + (p - 1.0)**2 ))    
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + (p - 1.0) * np.log(1.0 - x0**2)
    
    samples = np.zeros((N,1))
    
    # Loop over number of samples 
    for i in range(N):
        
        # Continue unil you have an acceptable sample 
        while True: 
            
            # Sample Beta distribution
            Z = np.random.beta( (p - 1.0)/2.0, (p - 1.0)/2.0 )
            
            # Sample Uniform distribution
            U = np.random.uniform(low=0.0,high=1.0)
            
            # W is essentially t
            W = (1.0 - (1.0 + b) * Z) / (1.0 - (1.0 - b) * Z)
            
            # Check whether to accept or reject 
            if kappa * W + (p - 1.0)*np.log(1.0 - x0*W) - c >= np.log(U):
                
                # Accept sample
                samples[i] = W
                break
                          
    return samples
                


def rand_von_mises_fisher(mu,kappa,N=1):
    """
        rand_von_mises_fisher(mu,kappa,N=1)
        ===================================
        
        Samples the von Mises-Fisher distribution with mean direction mu and concentration kappa. 
        
        INPUT: 
        
            * mu (array of floats of shape (p,1)) - mean direction. This should be a unit vector.
            * kappa (float) - concentration. 
            * N (int) - Number of samples. 
        
        OUTPUT: 
        
            * samples (array of floats of shape (N,p)) - samples of the von Mises-Fisher distribution
            with mean direction mu and concentration kappa. 
    """
    
    
    # Check that mu is a unit vector
    eps = 10**(-8) # Precision 
    norm_mu = np.linalg.norm(mu)
    if abs(norm_mu - 1.0) > eps:
        raise Exception("mu must be a unit vector.")
        
    # Check kappa >= 0 is numeric 
    if (kappa < 0) or ((type(kappa) is not float) and (type(kappa) is not int)):
        raise Exception("kappa must be a non-negative number.")
    
    # Check N>0 and is an int
    if (N<=0) or (type(N) is not int):
        raise Exception("N must be a non-zero positive integer.")
    
    # Dimension p
    p = len(mu)
    
    # Make sure that mu has a shape of px1
    mu = np.reshape(mu,(p,1))
    
    # Array to store samples 
    samples = np.zeros((N,p))
    
    #  Component in the direction of mu (Nx1)
    t = rand_t_marginal(kappa,p,N) 
    
    # Component orthogonal to mu (Nx(p-1))
    xi = rand_uniform_hypersphere(N,p-1) 
   
    # von-Mises-Fisher samples Nxp
    
    # Component in the direction of mu (Nx1).
    # Note that here we are choosing an 
    # intermediate mu = [1, 0, 0, 0, ..., 0] later
    # we rotate to the desired mu below
    samples[:,[0]] = t 
    
    # Component orthogonal to mu (Nx(p-1))
    samples[:,1:] = np.matlib.repmat(np.sqrt(1 - t**2), 1, p-1) * xi
    
    # Rotation of samples to desired mu
    O = null_space(mu.T)
    R = np.concatenate((mu,O),axis=1)
    samples = np.dot(R,samples.T).T
    
    return samples