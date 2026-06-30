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
    "f": {"value": 0.5, "limits": (0, 1), "fixed": False}, # Fading (0-1)
    "phi": {"value": 180*np.pi/180, "limits": (0, 2*np.pi), "fixed": False}, # Orientation angle (0-2pi radians)
    "gauss_blur_sigma": {"value": 4, "limits": (0, 10), "fixed": False}, # Gaussian blur sigma (in uas)
    "stretchx": {"value": 1., "limits": (0.5, 2), "fixed": True}, # Stretch term for the x-axis (default is 1)
    "stretchy": {"value": 1., "limits": (0.5, 2), "fixed": True}, # Stretch term for the y-axis (default is 1)
}

class sdisk(object):
    """Class for generating slashed disk models."""

    def __init__(self, I0=1, Rp=40, xoff=0, yoff=0,
                 dim=128, fov=225, f=0, phi=0,
                 gauss_blur_sigma=0, stretchx=1, stretchy=1):
        """Creates a slashed disk model.

        Args:
            I0 (float) : Total flux (Jy)
            Rp (float) : Radius (in uas)
            xoff (float) : x offset (in uas)
            yoff (float) : y offset (in uas)
            dim (int) : Dimensions of the image along an axis (square image)
            fov (int) : Field of view (in uas)
            f (float) : Fading factor (0,1)
            phi (float) : Slashing angle (in degrees)
            gauss_blur_sigma (float) : Gaussian blur sigma (in uas)
            stretchx (float) : Stretch term for the x-axis (default is 1)
            stretchy (float) : Stretch term for the y-axis (default is 1)
        Return:
            sdisk model with parameters
        """
        self.MODEL_PARAMS = MODEL_PARAMS
        self.model_name = 'sdisk'
        
        self.I0 = I0
        self.Rp = Rp
        self.xoff = xoff
        self.yoff = yoff
        self.dim = dim
        self.fov = fov
        self.f = 1-f
        self.phi = phi
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
        if utils._any_tensor([self.I0, self.Rp, self.xoff, self.yoff, self.f, self.phi, self.gauss_blur_sigma, self.stretchx, self.stretchy]):
            ref = next(v for v in [self.I0, self.Rp, self.xoff, self.yoff, self.f, self.phi, self.gauss_blur_sigma, self.stretchx, self.stretchy] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            phi = torch.as_tensor(self.phi, dtype=ref.dtype, device=ref.device)
            cos_phi = torch.cos(phi + torch.pi / 2)
            sin_phi = torch.sin(phi + torch.pi / 2)
            xoff, yoff = utils._as_tensors(self.xoff, self.yoff, dtype=ref.dtype, device=ref.device)
            Rp, I0, f = utils._as_tensors(self.Rp, self.I0, self.f, dtype=ref.dtype, device=ref.device)
            stretchx, stretchy = utils._as_tensors(self.stretchx, self.stretchy, dtype=ref.dtype, device=ref.device)
            x0 = (xx - xoff * stretchx) * cos_phi / stretchx + (yy - yoff * stretchy) * sin_phi / stretchy
            y0 = -(yy - yoff * stretchy) * cos_phi / stretchy + (xx - xoff * stretchx) * sin_phi / stretchx
            radius = torch.sqrt(x0**2 + (y0 / stretchy)**2)
            mask = imutils.soft_inside(radius, Rp, self.psize)
            sdisk_arr = ((1 - f) * y0 / Rp + 1 + f) * mask
            sdisk_arr = sdisk_arr / sdisk_arr.sum() * I0
            return imutils.gauss_blur(sdisk_arr, self.gauss_blur_sigma, self.psize)

        cos_phi = np.cos(self.phi+np.pi/2)
        sin_phi = np.sin(self.phi+np.pi/2)
        xx, yy = np.meshgrid(self.X, self.Y, indexing='xy')
        x0 = (xx - self.xoff * self.stretchx) * cos_phi / self.stretchx + (yy - self.yoff * self.stretchy) * sin_phi / self.stretchy
        y0 = -(yy - self.yoff * self.stretchy) * cos_phi / self.stretchy + (xx - self.xoff * self.stretchx) * sin_phi / self.stretchx
        r0 = x0**2 + y0**2
        sdisk_arr = np.zeros((self.dim, self.dim))
        mask = r0 < self.Rp**2
        sdisk_arr[mask] = ((1-self.f) * y0[mask] / self.Rp + 1 + self.f)
        sdisk_arr = sdisk_arr/np.sum(sdisk_arr)*self.I0
        sdisk_arr = imutils.gauss_blur(sdisk_arr, self.gauss_blur_sigma, self.psize) 
        return sdisk_arr

    
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
            if uv.ndim != 2 or (uv.shape[0] != 2 and uv.shape[1] != 2):
                raise ValueError("uv must have shape (2, N) or (N, 2).")
            if uv.shape[0] != 2:
                uv = uv.transpose(0, 1)

            phi = torch.as_tensor(self.phi, dtype=uv.dtype, device=uv.device)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            u_rot = uv[0] * cos_phi + uv[1] * sin_phi
            v_rot = -uv[0] * sin_phi + uv[1] * cos_phi
            uv_rot = torch.stack([u_rot, v_rot], dim=0)

            xoff = self.xoff * torch.cos(-phi) + self.yoff * torch.sin(-phi)
            yoff = -self.xoff * torch.sin(-phi) + self.yoff * torch.cos(-phi)
            stretch = torch.tensor([self.stretchx, self.stretchy], dtype=uv.dtype, device=uv.device)
            anaVis = vis.stretch_vis(uv_rot, lambda uv: vis.slashed_disk(uv, self.I0, self.Rp*1e-6/206265, 1-self.f, [-xoff*1e-6/206265, -yoff*1e-6/206265]), stretch)
            anaVis = vis.blur_by_gauss_kernel(uv_rot, anaVis, self.gauss_blur_sigma*1e-6/206265)
        return anaVis

    