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
    'xoff': {"value": 15, "limits": (-80, 80), "fixed": False}, # x offset (in uas)
    'yoff': {"value": -30, "limits": (-80, 80), "fixed": False}, # y offset (in uas)
    "gauss_blur_sigma": {"value": 4, "limits": (0, 10), "fixed": False} # Gaussian blur sigma (in uas)
}

class point(object):
    """Class for generating geometric point source models."""

    def __init__(self, I0=1, xoff=0, yoff=0,
                 dim=128, fov=225,
                 gauss_blur_sigma=0):
        """Creates a point source model.

        Args:
            I0 (float) : Total flux (Jy)
            xoff (float) : x offset (in uas)
            yoff (float) : y offset (in uas)
            dim (int) : Dimensions of the image along an axis (square image)
            fov (int) : Field of view (in uas)
            gauss_blur_sigma (float) : Gaussian blur sigma (in uas)

        Return:
            Point source model with parameters
        """
        self.MODEL_PARAMS = MODEL_PARAMS
        self.model_name = 'point'
        
        self.I0 = I0
        self.xoff = xoff
        self.yoff = yoff
        self.dim = dim
        self.fov = fov
        self.gauss_blur_sigma = gauss_blur_sigma

        self.X = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.Y = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.psize = (self.X[1]-self.X[0])
    
    def sky_map(self):
        """Generates the intensity map of the model
         
        Returns:
            Intensity map of the model
        """
        if utils._any_tensor([self.I0, self.xoff, self.yoff, self.gauss_blur_sigma]):
            ref = next(v for v in [self.I0, self.xoff, self.yoff, self.gauss_blur_sigma] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            xoff, yoff, gauss_blur_sigma, I0 = utils._as_tensors(self.xoff, self.yoff, self.gauss_blur_sigma, self.I0, dtype=ref.dtype, device=ref.device)
            model = torch.zeros_like(xx)
            if self.gauss_blur_sigma > 0:
                model += I0 * torch.exp(-((xx-xoff)**2 + (yy-yoff)**2) / (2 * gauss_blur_sigma**2))
            else:
                x_idx = torch.argmin(torch.abs(X - xoff))
                y_idx = torch.argmin(torch.abs(Y - yoff))
                model[y_idx, x_idx] = I0
            model = model / model.sum() * I0
            return model
        else:
            xx, yy = np.meshgrid(self.X, self.Y, indexing='xy')
            model = np.zeros_like(xx)
            if self.gauss_blur_sigma > 0:
                model += self.I0 * np.exp(-((xx-self.xoff)**2 + (yy-self.yoff)**2) / (2 * self.gauss_blur_sigma**2))
            else:
                x_idx = np.argmin(np.abs(self.X - self.xoff))
                y_idx = np.argmin(np.abs(self.Y - self.yoff))
                model[y_idx, x_idx] = self.I0
            model = model / model.sum() * self.I0
            return model

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
            anaVis = vis.point(uv, self.I0, offset=(self.xoff*1e-6/206265, self.yoff*1e-6/206265))
            anaVis = vis.blur_by_gauss_kernel(uv, anaVis, self.gauss_blur_sigma*1e-6/206265)
            return anaVis