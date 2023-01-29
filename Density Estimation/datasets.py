## From https://github.com/noamroze/moser_flow

import numpy as np
import torch
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
import trimesh
import igl
import scipy
from PIL import Image
import plotly.offline as offline

from torch.utils.data import Dataset


import importlib
if importlib.util.find_spec("ext_code") is not None:
    from ext_code import earth_plot
    
    
    
def coordinates_to_xyz(theta, phi):
    if isinstance(theta, torch.Tensor):
        lib = torch
        concatenate = torch.cat
    else:
        lib = np
        concatenate = np.concatenate

    x = lib.sin(theta) * lib.cos(phi)
    y = lib.sin(theta) * lib.sin(phi)
    z = lib.cos(theta)
    return concatenate([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], 1)

def xyz_to_coordinates(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    if isinstance(points, torch.Tensor):
        acos = torch.acos
        atan2 = torch.atan2
    else:
        acos = np.arccos
        atan2 = np.arctan2
    
    theta = acos(z)
    phi = atan2(y, x)
    return theta, phi

def latlon_to_xyz(data):
    theta = (90 - data[:, 0]) * np.pi / 180
    phi = (data[:, 1]) * np.pi / 180
    return coordinates_to_xyz(theta, phi)

def xyz_to_latlon(data_R):
    theta, phi = xyz_to_coordinates(data_R)
    xx = -(180/np.pi) * theta + 90
    yy = phi * 180/np.pi
    return xx, yy



class Dataset3D(Dataset):
    @staticmethod
    def fig_to_vtk(fig, mesh, out_dir, name):
        plots3D.colored_mesh_to_vtk(
            os.path.join(out_dir, "%s.vtk" %name),
            mesh,
            fig.data[0].intensity,
            name
        )

    def initial_plots(self, eval_dir, model, **kwargs):
        n_points = 100
        fig = plots3D.plot_histogram_on_surface(r"data", self[:len(self)][0].cpu().detach().numpy(), model.surface, n_points)
        offline.plot(fig, filename='{0}/data.html'.format(eval_dir), auto_open=False)
        self.fig_to_vtk(fig, model.surface.mesh, eval_dir, "data_samples")
        
        self.colorscale = (fig.data[0].cmin, fig.data[0].cmax)
        fig = plots3D.plot_histogram_on_surface(r"uniform_sample", model.monte_carlo_prior.sample((len(self), )).cpu().numpy(), model.surface, n_points)
        offline.plot(fig, filename='{0}/uniform_sample.html'.format(eval_dir), auto_open=False)

    def evaluate_model(self, model, epoch, eval_dir=None, logger=None, **args):
        n_points = 100
        def normalize_density(f):
            def wrapper(x):
                return f(x) / model.prior.normalizing_constant
            return wrapper
        fig = plots3D.plot_function_on_surface(r"$\nu-\nabla\cdot v$", normalize_density(model.signed_mu), model.surface, model.device, n_points, colorscale=self.colorscale)
        offline.plot(fig, filename='{0}/density.html'.format(eval_dir), auto_open=False)
        self.fig_to_vtk(fig, model.surface.mesh, eval_dir, "density")

        plt.close('all')

    def test_model(self, model, run_dir, *args, **kwargs):
        n_points = 100
        random_samples = model.prior.sample((len(self),))
        random_samples.requires_grad = True

        generated_samples = run_func_in_batches(model.transport, random_samples, 50000, 3)
        fig = plots3D.plot_histogram_on_surface(r"generated_samples", generated_samples.cpu().detach().numpy(), model.surface, n_points, colorscale=self.colorscale)
        offline.plot(fig, filename='{0}/generated_samples.html'.format(run_dir), auto_open=False)
        self.fig_to_vtk(fig, model.surface.mesh, run_dir, "generated_samples")
        

class EarthData(Dataset3D):
    def __init__(self, name, data, test_data=None):
        self.name = name
        self.data = self.latlon_to_xyz(data)
        if test_data is not None:
            self.test_data = self.latlon_to_xyz(test_data)
#         self.surface = get_surface("sphere")
        # self.surface.calc_mesh(100, (-1, 1))

    def latlon_to_xyz(self, data):
        theta = (90 - data[:, 0]) * np.pi / 180
        phi = (data[:, 1]) * np.pi / 180
        return coordinates_to_xyz(theta, phi)

    def __getitem__(self, index):
        return self.data[index], 0

    def __len__(self):
        return len(self.data)

    def initial_plots(self, eval_dir, **kwargs):
        self.colorscale = None

    def evaluate_model(self, model, epoch, eval_dir=None, logger=None, **args):
        if "earth_plot" in globals():
            figs = earth_plot(self.name, model, torch.tensor(self.data, device=model.device), torch.tensor(self.test_data, device=model.device), model.device, 200)
            for i, fig in enumerate(figs):
                fig.savefig(os.path.join(eval_dir, "earth_plot_%s.png" %(i+1)))
                fig.savefig(os.path.join(eval_dir, "earth_plot_%s.pdf"%(i+1)))

    def test_model(self, model, run_dir, *args, **kwargs):
        pass
    
    
    
class DataHandler:
    def __init__(self, config, eps):
        training_size = config["training_size"]
        validation_size = config["validation_size"]
        test_size = config["test_size"]
        self.training_set = self.create_dataset(config, training_size, eps)
        self.validation_set = self.create_dataset(config, validation_size, eps)
        self.test_set = self.create_dataset(config, test_size, eps)

    def create_dataset(self, config, size, eps):
        raise NotImplementedError

    def get_dataloaders(self, batch_size, eval_batch_size, **kwargs):
        datasets = [self.training_set, self.validation_set, self.test_set]
        batch_sizes = [batch_size, eval_batch_size, eval_batch_size]
        return (torch.utils.data.DataLoader(dataset, batch_size=batch_size, **kwargs) for dataset, batch_size in zip(datasets, batch_sizes))


class EarthDataHandler(DataHandler):
    def __init__(self, config, eps):
        csv_path = "./data/earth_data/%s.csv" % config["name"]
        data = pd.read_csv(csv_path, comment="#", header=0).values.astype("float32")

        training_size = config["training_size"]
        validation_size = config["validation_size"]
        test_size = config["test_size"]
        total_config_size = training_size + validation_size + test_size
        training_size = int((training_size / total_config_size) * len(data))
        validation_size = int((validation_size / total_config_size) * len(data))
        test_size = len(data) - validation_size - training_size

        train_data, val_data, test_data = torch.utils.data.random_split(data, [training_size, validation_size, test_size])
        if validation_size == 0:
            val_data = test_data
        self.training_set = EarthData(config["name"], data[train_data.indices], data[test_data.indices]) 
        self.validation_set = EarthData(config["name"], data[val_data.indices])
        self.test_set = EarthData(config["name"], data[test_data.indices])

