import torch
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d.art3d import Poly3DCollection



theta = np.linspace(0, 2 * np.pi, 2 * 100)
phi = np.linspace(0, np.pi, 100)
tp = np.array(np.meshgrid(theta, phi, indexing='ij'))
tp = tp.transpose([1, 2, 0]).reshape(-1, 2)

def spherical_to_euclidean(sph_coords):
    """
        https://github.com/katalinic/sdflows/blob/0f319d8ae6e2c858061a0a31880d4b70f69b6a64/utils.py#L4
    """
    if sph_coords.ndim == 1:
        sph_coords = np.expand_dims(sph_coords, 0)
        
    theta, phi = np.split(sph_coords, 2, 1)
    return np.concatenate((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ), 1)


def spherical_to_euclidean_torch(x):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    
    theta = x[:,0]
    phi = x[:,1]
    
    xx = torch.sin(phi)*torch.cos(theta)
    yy = torch.sin(phi)*torch.sin(theta)
    zz = torch.cos(phi)
    
    return torch.cat([xx[:,None],yy[:,None],zz[:,None]], dim=-1)


def euclidean_to_spherical(euc_coords):
    """
        https://github.com/katalinic/sdflows/blob/0f319d8ae6e2c858061a0a31880d4b70f69b6a64/utils.py#L15
    """
    if euc_coords.ndim == 1:
        euc_coords = np.expand_dims(euc_coords, 0)
    x, y, z = np.split(euc_coords, 3, 1)
    return np.concatenate((
        np.pi + np.arctan2(-y, -x),
        np.arccos(z)
    ), 1)


def _plot_mollweide(heatmap):
    """
        https://github.com/katalinic/sdflows/blob/0f319d8ae6e2c858061a0a31880d4b70f69b6a64/plotting.py
    """
    tt, pp = np.meshgrid(theta - np.pi, phi - np.pi / 2, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.pcolormesh(tt, pp, heatmap, cmap=plt.cm.jet)
    ax.set_axis_off()
    plt.show()
    
    
def plot_target_density(target_fn):    
    density = target_fn(spherical_to_euclidean(tp))
    heatmap = density.reshape(2 * 100, 100)
    _plot_mollweide(heatmap)
    

def scatter_mollweide(X_target, target_fn):
    density = target_fn(spherical_to_euclidean(tp))
    heatmap = density.reshape(2 * 100, 100)
    
    tt, pp = np.meshgrid(theta - np.pi, phi - np.pi / 2, indexing='ij')

    spherical_coords = euclidean_to_spherical(X_target)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.pcolormesh(tt, pp, heatmap, cmap=plt.cm.jet)
    ax.scatter(spherical_coords[:,0]-np.pi, spherical_coords[:,1]-np.pi/2)
    
    ax.set_axis_off()
    plt.show()

    
def projection_mollweide(target_fn, ax, vmax=None):
    density = target_fn(spherical_to_euclidean(tp))
    heatmap = density.reshape(2 * 100, 100)
    tt, pp = np.meshgrid(theta - np.pi, phi - np.pi / 2, indexing='ij')
    ax.pcolormesh(tt, pp, heatmap, cmap=plt.cm.jet,vmax=vmax)
    ax.set_axis_off()
    
    
def scatter_mollweide_ax(X_target, ax):    
    spherical_coords = euclidean_to_spherical(X_target)
    ax.scatter(spherical_coords[:,0]-np.pi, spherical_coords[:,1]-np.pi/2)
    ax.set_axis_off()


def plot_3d_scatter(data, ax=None, colour='red', sz=30, el=20, az=50, sph=True, sph_colour="gray", 
                    sph_alpha=0.03, eq_line=False, pol_line=False, grd=False):
    """
        From https://github.com/dlwhittenbury/von-Mises-Fisher-Sampling/blob/master/von-Mises-Fisher.ipynb
        
        plot_3d_scatter()
        =================
        
        Plots 3D samples on the surface of a sphere.
        
        INPUT: 
        
            * data (array of floats of shape (N,3)) - samples of a spherical distribution such as von Mises-Fisher.
            * ax (axes) - axes on which the plot is constructed.
            * colour (string) - colour of the scatter plot.
            * sz (float) - size of points.
            * el (float) - elevation angle of the plot.
            * az (float) - azimuthal angle of the plot.
            * sph (boolean) - whether or not to inclde a sphere.
            * sph_colour (string) - colour of the sphere if included.
            * sph_alpha (float) - the opacity/alpha value of the sphere.
            * eq_line (boolean) - whether or not to include an equatorial line.
            * pol_line (boolean) - whether or not to include a polar line.
            * grd (boolean) - whether or not to include a grid.
            
        OUTPUT: 
        
            * ax (axes) - axes on which the plot is contructed. 
            * Plot of 3D samples on the surface of a sphere. 
            
    """
    
    
    # The polar axis
    if ax is None:
        ax = plt.axes(projection='3d')
        
    # Check that data is 3D (data should be Nx3)
    d = np.shape(data)[1]
    if d != 3: 
        raise Exception("data should be of shape Nx3, i.e., each data point should be 3D.")
        
    ax.scatter(data[:,0],data[:,1],data[:,2],s=10,c=colour,cmap=plt.cm.Spectral)
    ax.view_init(el, az)
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_zlim(-1.5,1.5)
    
    # Add a shaded unit sphere
    if sph:
        u, v = np.mgrid[0:2*np.pi:30j, 0:np.pi:30j]
        x = np.cos(u)*np.sin(v)
        y = np.sin(u)*np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color=sph_colour,alpha=sph_alpha)
        ax.plot_wireframe(x, y, z, linewidth=1, alpha=0.25, color="gray")

    
    # Add an equitorial line 
    if eq_line: 
        # t = theta, p = phi
        eqt = np.linspace(0,2*np.pi,50,endpoint=False)
        eqp = np.linspace(0,2*np.pi,50,endpoint=False)
        eqx = 2*np.sin(eqt)*np.cos(eqp) 
        eqy = 2*np.sin(eqt)*np.sin(eqp) - 1
        eqz = np.zeros(50)
        
        # Equator line
        ax.plot(eqx,eqy,eqz,color="k",lw=1)
       
    # Add a polar line 
    if pol_line: 
        # t = theta, p = phi
        eqt = np.linspace(0,2*np.pi,50,endpoint=False)
        eqp = np.linspace(0,2*np.pi,50,endpoint=False)
        eqx = 2*np.sin(eqt)*np.cos(eqp) 
        eqy = 2*np.sin(eqt)*np.sin(eqp) - 1
        eqz = np.zeros(50)
        
        # Polar line
        ax.plot(eqx,eqz,eqy,color="k",lw=1)
        
    # Draw a centre point
#     ax.scatter([0], [0], [0], color="k", s=sz)    
        
    # Turn off grid 
    ax.grid(grd)
    
    # Ticks 
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    ax.set_zticks([-1,0,1])
    
    return ax


def fill_between_3d(ax,x1,y1,z1,x2,y2,z2,mode=1,c='steelblue',alpha=0.6):
    
    """
    From https://github.com/artmenlope/matplotlib-fill_between-in-3D/blob/master/FillBetween3d.py
    
    Function similar to the matplotlib.pyplot.fill_between function but 
    for 3D plots.
       
    input:
        
        ax -> The axis where the function will plot.
        
        x1 -> 1D array. x coordinates of the first line.
        y1 -> 1D array. y coordinates of the first line.
        z1 -> 1D array. z coordinates of the first line.
        
        x2 -> 1D array. x coordinates of the second line.
        y2 -> 1D array. y coordinates of the second line.
        z2 -> 1D array. z coordinates of the second line.
    
    modes:
        mode = 1 -> Fill between the lines using the shortest distance between 
                    both. Makes a lot of single trapezoids in the diagonals 
                    between lines and then adds them into a single collection.
                    
        mode = 2 -> Uses the lines as the edges of one only 3d polygon.
           
    Other parameters (for matplotlib): 
        
        c -> the color of the polygon collection.
        alpha -> transparency of the polygon collection.
        
    """

    if mode == 1:
        
        for i in range(len(x1)-1):
            
            verts = [(x1[i],y1[i],z1[i]), (x1[i+1],y1[i+1],z1[i+1])] + \
                    [(x2[i+1],y2[i+1],z2[i+1]), (x2[i],y2[i],z2[i])]
            
            ax.add_collection3d(Poly3DCollection([verts],
                                                 alpha=alpha,
                                                 linewidths=0,
                                                 color=c))

    if mode == 2:
        
        verts = [(x1[i],y1[i],z1[i]) for i in range(len(x1))] + \
                [(x2[i],y2[i],z2[i]) for i in range(len(x2))]
                
        ax.add_collection3d(Poly3DCollection([verts],alpha=alpha,color=c))



def polar2cartesian(r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)    
    return x,y
    
def plot_angular_density_3d(theta,pdf,ax,colour="blue",axes="off",fs=16,label=None):
    
    """
        From https://github.com/artmenlope/matplotlib-fill_between-in-3D/blob/master/FillBetween3d.py
    
        plot_angular_density_3d(theta,pdf,colour="blue",axes="off",fs=16)
        ============================================================================
        
        Plots the probability density function of a circular distribution on the unit 
        circle.
        
        INPUT: 
        
            * theta - angular grid - an array of floats. 
            * pdf - the values of the probability density function on the angular grid theta. 
            * ax - figure axis
            * colour - an optional argument, the colour of the pdf curve, a string. 
            * axes - an optional argument, whether or not to include the axes, boolean.
            * fs - an optional argument, the fontsize.
        
        OUTPUT: 
            
            * ax (axes) - Return axes of plot 
            * A plot on a circle of a circular distribution.
    
    """
    
    # Draw the unit circle 
    # ====================
    
    # Radius
    r = 1.0
       
    # Convert polar coordinates to cartesian coordinates
    [x,y] = polar2cartesian(r,theta)
    
    # Plot the unit circle 
   # plt.plot(x,y,0,color='black',lw=2,zorder=0)
    
    # Draw angular probability density 
    # ================================
        
    # Convert polar coordinates to cartesian coordinates
    [xi,yi] = polar2cartesian(r,theta)
    
    z = pdf
    
    # Plot the PDF
    ax.plot(xi,yi,z,color=colour,lw=2,label=label)
    
    #fill beetween
    fill_between_3d(ax,x,y,[0]*len(theta),xi,yi,z,mode=1,c=colour,alpha=0.4)

    # Limits
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    
    # Turn off grid 
    ax.grid(False)
    
    # Labels 
    ax.set_xlabel('x',fontsize=fs)
    ax.set_ylabel('y',fontsize=fs)
    ax.set_zlabel('z',fontsize=fs)
    ax.zaxis.set_rotate_label(False)
    
    # Ticks 
    ax.set_xticks([-1,0,1])
    ax.set_yticks([-1,0,1])
    
    return ax