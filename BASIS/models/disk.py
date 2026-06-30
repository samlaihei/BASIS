from __future__ import division
from __future__ import print_function

from builtins import object
import numpy as np
import torch
import BASIS.modules.imutils as imutils
import BASIS.modules.vis as vis
import BASIS.modules.utils as utils

MODEL_PARAMS = {
    'I0': {"value": 1, "limits": (0.05, 10), "fixed": False}, # Total flux density (Jy)
    'Rp': {"value": 24, "limits": (10, 40), "fixed": False}, # Radius (in uas)
    'xoff': {"value": 15, "limits": (-80, 80), "fixed": False}, # x offset (in uas)
    'yoff': {"value": -30, "limits": (-80, 80), "fixed": False}, # y offset (in uas)
    "gauss_blur_sigma": {"value": 4, "limits": (0, 10), "fixed": False}, # Gaussian blur sigma (in uas)
    "stretchx": {"value": 1., "limits": (0.5, 2), "fixed": True}, # Stretch term for the x-axis (default is 1)
    "stretchy": {"value": 1., "limits": (0.5, 2), "fixed": True} # Stretch term for the y-axis (default is 1)
}

class disk(object):
    """Class for generating geometric disk models."""

    def __init__(self, I0=1, Rp=40, xoff=0, yoff=0,
                 dim=128, fov=225,
                 gauss_blur_sigma=0, 
                 stretchx=1, stretchy=1):
        """Creates a disk model.

        Args:
            I0 (float) : Total flux (Jy)
            Rp (float) : Radius (in uas)
            xoff (float) : x offset (in uas)
            yoff (float) : y offset (in uas)
            dim (int) : Dimensions of the image along an axis (square image)
            fov (int) : Field of view (in uas)
            gauss_blur_sigma (float) : Gaussian blur sigma (in uas)
            stretchx (float) : Stretch term for the x-axis (default is 1)
            stretchy (float) : Stretch term for the y-axis (default is 1)

        Return:
            disk model with parameters
        """
        self.MODEL_PARAMS = MODEL_PARAMS
        self.model_name = 'disk'
        
        self.I0 = I0
        self.Rp = Rp
        self.xoff = xoff
        self.yoff = yoff
        self.dim = dim
        self.fov = fov
        self.gauss_blur_sigma = gauss_blur_sigma
        self.stretchx = stretchx
        self.stretchy = stretchy
        self.X = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.Y = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.psize = (self.X[1]-self.X[0])
    
    def sky_map(self):
        """Generates the intensity map of the model
         
        Returns:
            Intensity map of the model
        """
        if utils._any_tensor([self.I0, self.Rp, self.xoff, self.yoff, self.gauss_blur_sigma, self.stretchx, self.stretchy]):
            ref = next(v for v in [self.I0, self.Rp, self.xoff, self.yoff, self.gauss_blur_sigma, self.stretchx, self.stretchy] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            xoff, yoff = utils._as_tensors(self.xoff, self.yoff, dtype=ref.dtype, device=ref.device)
            Rp, I0, stretchx, stretchy = utils._as_tensors(self.Rp, self.I0, self.stretchx, self.stretchy, dtype=ref.dtype, device=ref.device)
            radius = torch.sqrt(((xx - xoff) / stretchx) ** 2 + ((yy - yoff) / stretchy) ** 2)
            disk_arr = imutils.soft_inside(radius, Rp, self.psize)
            disk_arr = disk_arr / disk_arr.sum() * I0
            return imutils.gauss_blur(disk_arr, self.gauss_blur_sigma, self.psize)

        disk_arr = np.zeros((self.dim, self.dim))
        xx, yy = np.meshgrid(self.X, self.Y)
        r2 = ((xx - self.xoff * self.stretchx) / self.stretchx) ** 2 + ((yy - self.yoff * self.stretchy) / self.stretchy) ** 2
        disk_arr = (r2 <= self.Rp ** 2).astype(float)
        disk_arr = disk_arr / np.sum(disk_arr) * self.I0
        disk_arr = imutils.gauss_blur(disk_arr, self.gauss_blur_sigma, self.psize)
        return disk_arr

    
    def sample_vis(self, uv, ttype='analytical'):
        """Samples the visibilities at given (u, v) coordinates.

        Parameters
        ----------
        uv : array_like
            The (u, v) coordinates in wavelengths.
        ttype : str
            The type of sampling method ('analytical' or 'direct').

        Returns
        -------
        vis : array_like
            The sampled visibilities.
        """
        if ttype == 'direct' or ttype=='DFT':
            image = self.sky_map()
            dvis = vis.DFT(image.unsqueeze(0) if torch.is_tensor(image) else np.expand_dims(image, 0), uv, xfov=self.fov, yfov=self.fov)[0]
            return dvis

        else:
            uv = torch.as_tensor(uv)
            stretch = torch.tensor([self.stretchx, self.stretchy], dtype=uv.dtype, device=uv.device)
            anaVis = vis.stretch_vis(uv, lambda uv: vis.disk(uv, self.I0, self.Rp*1e-6/206265, [-self.xoff*1e-6/206265, -self.yoff*1e-6/206265]), stretch)
            anaVis = vis.blur_by_gauss_kernel(uv, anaVis, self.gauss_blur_sigma*1e-6/206265)
        return anaVis

    