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
    'gauss_sigma': {"value": 20, "limits": (5, 40), "fixed": False}, # Gaussian sigma (in uas)
    'aq': {"value": 0.4, "limits": (0.05, 1), "fixed": False}, # Axis ratio (0,1)
    "pa": {"value":20*np.pi/180, "limits": (0, np.pi), "fixed": False}, # Position angle (0-pi radians)
    'xoff': {"value": 15, "limits": (-80, 80), "fixed": False}, # x offset (in uas)
    'yoff': {"value": -30, "limits": (-80, 80), "fixed": False}, # y offset (in uas)
    'gauss_blur_sigma': {"value": 4, "limits": (0, 10), "fixed": False}, # Gaussian blur sigma (in uas)
    'stretchx': {"value": 1., "limits": (0.5, 2), "fixed": True}, # Stretch term for the x-axis (default is 1)
    'stretchy': {"value": 1., "limits": (0.5, 2), "fixed": True}, # Stretch term for the y-axis (default is 1)
}

class gauss(object):
    """Class for generating Gaussian models."""

    def __init__(self, I0=1, gauss_sigma=20, aq=0.4, pa=20*np.pi/180.0, xoff=0, yoff=0,
                 dim=128, fov=225,
                 gauss_blur_sigma=0, 
                 stretchx=1.0, stretchy=1.0):
        """Creates a Gaussian model.

        Args:
            I0 (float) : Total flux (Jy)
            gauss_sigma (float) : Gaussian sigma (in uas)
            aq (float) : Axis ratio (0,1)
            pa (float) : Position angle (0-pi radians)
            xoff (float) : x offset (in uas)
            yoff (float) : y offset (in uas)
            dim (int) : Dimensions of the image along an axis (square image)
            fov (int) : Field of view (in uas)
            gauss_blur_sigma (float) : Gaussian blur sigma (in uas)
            stretchx (float) : Stretch term for the x-axis (default is 1)
            stretchy (float) : Stretch term for the y-axis (default is 1)

        Return:
            Gaussian model with parameters
        """
        self.MODEL_PARAMS = MODEL_PARAMS
        self.model_name = 'gauss'
        
        self.I0 = I0
        self.gauss_sigma = gauss_sigma
        self.aq = aq
        self.pa = pa  
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
        if utils._any_tensor([self.I0, self.gauss_sigma, self.aq, self.pa, self.xoff, self.yoff, self.gauss_blur_sigma, self.stretchx, self.stretchy]):
            ref = next(v for v in [self.I0, self.gauss_sigma, self.aq, self.pa, self.xoff, self.yoff, self.gauss_blur_sigma, self.stretchx, self.stretchy] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            pa = torch.as_tensor(self.pa, dtype=ref.dtype, device=ref.device)
            cos_pa = torch.cos(pa)
            sin_pa = torch.sin(pa)
            xoff, yoff = utils._as_tensors(self.xoff, self.yoff, dtype=ref.dtype, device=ref.device)
            aq, gauss_sigma, I0 = utils._as_tensors(self.aq, self.gauss_sigma, self.I0, dtype=ref.dtype, device=ref.device)
            stretchx, stretchy = utils._as_tensors(self.stretchx, self.stretchy, dtype=ref.dtype, device=ref.device)
            y0 = xx * cos_pa / stretchx - yy * sin_pa / stretchy - xoff * cos_pa + yoff * sin_pa
            x0 = yy * cos_pa / stretchy + xx * sin_pa / stretchx - yoff * cos_pa - xoff * sin_pa
            r0 = x0 ** 2 + (y0 / aq) ** 2
            gauss_arr = torch.exp(-r0 / (2. * gauss_sigma ** 2))
            gauss_arr = gauss_arr / gauss_arr.sum() * I0
            return imutils.gauss_blur(gauss_arr, self.gauss_blur_sigma, self.psize)

        cos_pa = np.cos(self.pa)
        sin_pa = np.sin(self.pa)
        xx, yy = np.meshgrid(self.X, self.Y, indexing='xy')
        y0 = xx * cos_pa / self.stretchx - yy * sin_pa / self.stretchy - self.xoff * cos_pa + self.yoff * sin_pa
        x0 = yy * cos_pa / self.stretchy + xx * sin_pa / self.stretchx - self.yoff * cos_pa - self.xoff * sin_pa
        r0 = x0 ** 2 + (y0 / self.aq) ** 2
        gauss_arr = np.exp(-r0 / (2. * self.gauss_sigma ** 2))
        gauss_arr = gauss_arr/np.sum(gauss_arr)*self.I0

        gauss_arr = imutils.gauss_blur(gauss_arr, self.gauss_blur_sigma, self.psize)
        return gauss_arr
    
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
        if ttype == 'direct' or ttype=='DFT' or ttype=='dft':
            image = self.sky_map()
            dvis = vis.DFT(image.unsqueeze(0) if torch.is_tensor(image) else np.expand_dims(image, 0), uv, xfov=self.fov, yfov=self.fov)[0]
            return dvis

        else:
            uv = torch.as_tensor(uv)
            stretch = torch.tensor([self.stretchx, self.stretchy], dtype=uv.dtype, device=uv.device)
            anaVis = vis.stretch_vis(uv, lambda uv: vis.elliptical_gauss(uv, self.I0, self.aq*self.gauss_sigma*1e-6/206265, self.gauss_sigma*1e-6/206265, -self.pa, offset=(-self.xoff*1e-6/206265, -self.yoff*1e-6/206265)), stretch)
            anaVis = vis.blur_by_gauss_kernel(uv, anaVis, self.gauss_blur_sigma*1e-6/206265)
            return anaVis