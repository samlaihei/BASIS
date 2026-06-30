from __future__ import division
from __future__ import print_function

from builtins import object
import numpy as np
import torch
import GCFit.modules.imutils as imutils
import GCFit.modules.vis as vis
import GCFit.modules.utils as utils

MODEL_PARAMS = {
    "I0": {"value": 1, "limits": (0.05, 10), "fixed": False}, # Total flux density (Jy)
    "Rp": {"value": 24, "limits": (10, 40), "fixed": False}, # Outer radius (uas)
    "Rn_frac": {"value": 0.75, "limits": (0.05, 0.95), "fixed": False}, # Inner radius fraction (0-1)
    "ecn": {"value": 0.8, "limits": (0, 0.99), "fixed": False}, # Eccentricity (0-1)
    "f": {"value": 0.5, "limits": (0, 1), "fixed": False}, # Fading (0-1)
    "phi": {"value": 180*np.pi/180, "limits": (0, 2*np.pi), "fixed": False}, # Orientation angle (0-2pi radians)
    "gax": {"value": 0.8, "limits": (0.05, 2), "fixed": False}, # FWHM main axis ratio (0-inf)
    "aq": {"value": 0.4, "limits": (0.05, 1), "fixed": False}, # Axial ratio (0.01-1)
    "gq": {"value": 0.4, "limits": (0, 0.99), "fixed": False}, # Gaussian flux fraction (0-0.99)
    "floor_sigma": {"value": 30, "limits": (1, 100), "fixed": False}, # Floor gauss sigma (0-inf uas)
    "bg_gauss_flux": {"value": 0.4, "limits": (0, 2), "fixed": False}, # Background Gaussian flux (Jy)
    "bg_gauss_sigma": {"value": 48, "limits": (1, 100), "fixed": False}, # Background Gaussian sigma (uas)
    "floor_brightness": {"value": 0.05, "limits": (0, 2), "fixed": False}, # Floor brightness (Jy)
    "gauss_blur_sigma": {"value": 4, "limits": (0, 10), "fixed": False}, # Gaussian blur sigma (uas)
    "stretchx": {"value": 1.0, "limits": (0.5, 2), "fixed": True}, # Stretch term for the x-axis (0.5-2)
    "stretchy": {"value": 1.0, "limits": (0.5, 2), "fixed": True}, # Stretch term for the y-axis (0.5-2)
}

def _bessel_i(n, x):
    if n == 0:
        return torch.special.i0(x)
    if n == 1:
        return torch.special.i1(x)
    i_nm1 = torch.special.i0(x)
    i_n = torch.special.i1(x)
    safe_x = torch.where(x == 0, torch.ones_like(x), x)
    for k in range(1, n):
        i_np1 = i_nm1 - (2.0 * k / safe_x) * i_n
        i_np1 = torch.where(x == 0, torch.zeros_like(i_np1), i_np1)
        i_nm1, i_n = i_n, i_np1
    return i_n

class xsringauss(object):
    """Class for centered xsringauss Model.
    Reference: Benkevitch, L., Akiyama, K., Lu, R., Doeleman, S., & Fish, V. 2016,
    arXiv:1609.00055
    """

    def __init__(self, I0=1, Rp=42, Rn_frac=0.75, ecn=0, f=0, gax=0.5, aq=0.5, gq=0.1, phi=0, 
                 stretchx=1.0, stretchy=1.0,
                 fov=225, dim=512,
                 bg_gauss_flux=0, bg_gauss_sigma=50,
                 floor_brightness=0, floor_sigma=0, gauss_blur_sigma=0):
        """Creates a xsringauss Model.
        
        Args:
            I0 (float) : Total flux (Jy)
            Rp (float) : Outer radius (in uas)
            Rn_frac (float) : Inner radius fraction (0.1-0.9)
            ecn (float) : Eccentricity (0,1)
            f (float) : Fading parameter (0,1)
            gax (float) : FWHM of main axis
            aq (float) : Axial ratio
            gq (float) : Fraction of gaussian flux compared to total flux (0,1)
            phi (float) : Orientation
            stretchx (float) : Stretch term for the x-axis (0.5-2)
            stretchy (float) : Stretch term for the y-axis (0.5-2)
            fov (int) : Field of view (in uas)
            bg_gauss_flux (float) : Background Gaussian flux (Jy)
            bg_gauss_sigma (float) : Background Gaussian sigma (in uas)
            dim (int) : Dimensions of the image along an axis (square image)
            floor_brightness (float) : Floor brightness level (Jy)
            floor_sigma (float) : Standard deviation of floor gauss (in uas)
            gauss_blur_sigma (float) : Gaussian blur sigma (in uas)

        Return:
            xsringauss model with parameters
        """
        self.MODEL_PARAMS = MODEL_PARAMS
        self.model_name = 'xsringauss'

        self.I0 = I0
        self.Rp = Rp
        self.Rn = Rn_frac * Rp
        self.ecn = -ecn
        self.f = 1-f
        self.gax = gax
        self.aq =  aq
        self.gq = gq
        self.phi = phi - np.pi/4
        self.fov = fov
        self.dim = dim
        self.bg_gauss_flux = bg_gauss_flux
        self.bg_gauss_sigma = bg_gauss_sigma
        self.floor_brightness = floor_brightness
        self.floor_sigma = floor_sigma
        self.gauss_blur_sigma = gauss_blur_sigma
        self.stretchx = stretchx
        self.stretchy = stretchy
        self.V0 = I0*(1-self.gq) + floor_brightness
        self.V1 = gq*I0

        self.d = self.ecn * (self.Rp - self.Rn)
        self.a = self.gax * self.Rp
        self.b = self.aq*self.a
        
        self.X = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.Y = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.psize = (self.X[1]-self.X[0])
    
    def floor_component(self):
        """Generates the floor component of the model
                
        Returns:
            Floor component of the model
        """
        if utils._any_tensor([self.floor_brightness, self.floor_sigma, self.phi, self.d]):
            ref = next(v for v in [self.floor_brightness, self.floor_sigma, self.phi, self.d] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            phi, d, floor_sigma, floor_brightness = utils._as_tensors(self.phi, self.d, self.floor_sigma, self.floor_brightness, dtype=ref.dtype, device=ref.device)
            stretchx, stretchy = utils._as_tensors(self.stretchx, self.stretchy, dtype=ref.dtype, device=ref.device)
            cos_phi = torch.cos(phi+torch.pi/2)
            sin_phi = torch.sin(phi+torch.pi/2)
            x0 = xx * cos_phi / stretchx + yy * sin_phi / stretchy
            y0 = -yy * cos_phi / stretchy + xx * sin_phi / stretchx
            R = torch.sqrt((x0 - d / torch.sqrt(torch.tensor(2.0, dtype=ref.dtype, device=ref.device)))**2 + (y0 - d / torch.sqrt(torch.tensor(2.0, dtype=ref.dtype, device=ref.device)))**2)
            floor_arr = torch.exp(-R**2/(2.*floor_sigma**2))
            floor_arr = floor_arr / floor_arr.sum() * floor_brightness
            return floor_arr
        cos_phi = np.cos(self.phi+np.pi/2)
        sin_phi = np.sin(self.phi+np.pi/2)
        xx, yy = np.meshgrid(self.X, self.Y, indexing='xy')
        x0 = xx * cos_phi / self.stretchx + yy * sin_phi / self.stretchy
        y0 = -yy * cos_phi / self.stretchy + xx * sin_phi / self.stretchx
        R = np.sqrt((x0 - self.d / np.sqrt(2))**2 + (y0 - self.d / np.sqrt(2))**2)
        floor_arr = np.exp(-R**2/(2.*self.floor_sigma**2))
        floor_arr = floor_arr/np.sum(floor_arr)*self.floor_brightness
        return floor_arr
    
    def bg_gauss_component(self):
        """Generate the background Gaussian component of the model

        Returns:
            Background Gaussian component of the model
        """
        if utils._any_tensor([self.bg_gauss_flux, self.bg_gauss_sigma, self.phi]):
            ref = next(v for v in [self.bg_gauss_flux, self.bg_gauss_sigma, self.phi] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            phi, bg_gauss_flux, bg_gauss_sigma = utils._as_tensors(self.phi, self.bg_gauss_flux, self.bg_gauss_sigma, dtype=ref.dtype, device=ref.device)
            cos_phi = torch.cos(phi+torch.pi/2)
            sin_phi = torch.sin(phi+torch.pi/2)
            stretchx, stretchy = utils._as_tensors(self.stretchx, self.stretchy, dtype=ref.dtype, device=ref.device)
            x0 = xx * cos_phi / stretchx + yy * sin_phi / stretchy
            y0 = -yy * cos_phi / stretchy + xx * sin_phi / stretchx
            bg_gauss_arr = torch.exp(-(x0**2 + y0**2)/(2.*bg_gauss_sigma**2))
            bg_gauss_arr = bg_gauss_arr / bg_gauss_arr.sum() * bg_gauss_flux
            return bg_gauss_arr
        cos_phi = np.cos(self.phi+np.pi/2)
        sin_phi = np.sin(self.phi+np.pi/2)
        xx, yy = np.meshgrid(self.X, self.Y, indexing='xy')
        x0 = xx * cos_phi / self.stretchx + yy * sin_phi / self.stretchy
        y0 = -yy * cos_phi / self.stretchy + xx * sin_phi / self.stretchx
        bg_gauss_arr = np.exp(-(x0**2 + y0**2)/(2.*self.bg_gauss_sigma**2))
        bg_gauss_arr = bg_gauss_arr/np.sum(bg_gauss_arr)*self.bg_gauss_flux
        return bg_gauss_arr
        
    def sky_map(self):
        """Generates the intensity map of the model
         
        Returns:
            Intensity map of the model
        """
        if utils._any_tensor([self.I0, self.Rp, self.Rn, self.ecn, self.f, self.gax, self.aq, self.gq, self.phi, self.bg_gauss_flux, self.bg_gauss_sigma, self.floor_brightness, self.floor_sigma, self.gauss_blur_sigma]):
            ref = next(v for v in [self.I0, self.Rp, self.Rn, self.ecn, self.f, self.gax, self.aq, self.gq, self.phi, self.bg_gauss_flux, self.bg_gauss_sigma, self.floor_brightness, self.floor_sigma, self.gauss_blur_sigma] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            Rp, Rn, I0, f, gq, d = utils._as_tensors(self.Rp, self.Rn, self.I0, self.f, self.gq, self.d, dtype=ref.dtype, device=ref.device)
            a, b = utils._as_tensors(self.a, self.b, dtype=ref.dtype, device=ref.device)
            phi = torch.as_tensor(self.phi, dtype=ref.dtype, device=ref.device)
            cos_phi = torch.cos(phi+torch.pi/2)
            sin_phi = torch.sin(phi+torch.pi/2)
            stretchx, stretchy = utils._as_tensors(self.stretchx, self.stretchy, dtype=ref.dtype, device=ref.device)
            x0 = xx * cos_phi / stretchx + yy * sin_phi / stretchy
            y0 = -yy * cos_phi / stretchy + xx * sin_phi / stretchx
            slash_axis = (x0 + y0) / torch.sqrt(torch.tensor(2.0, dtype=ref.dtype, device=ref.device))
            half_diag = torch.sqrt(torch.tensor(2.0, dtype=ref.dtype, device=ref.device))
            r_inner = torch.sqrt((x0 - d / half_diag)**2 + (y0 - d / half_diag)**2)
            r_outer = torch.sqrt(x0**2 + y0**2)
            mask = imutils.soft_outside(r_inner, Rn, self.psize) * imutils.soft_inside(r_outer, Rp, self.psize)
            xs0 = (2 * I0 * (1 - gq) / torch.pi) / (Rp**2 - Rn**2 * (1 + d / Rp) - (1 - f) * (d * Rn**2 / Rp))
            xsringauss_arr = xs0 * ((1 - f) * slash_axis / Rp + 1 + f) * 0.5 * mask
            dx = x0 - (d + Rn) / torch.sqrt(torch.tensor(2.0, dtype=ref.dtype, device=ref.device))
            dy = y0 - (d + Rn) / torch.sqrt(torch.tensor(2.0, dtype=ref.dtype, device=ref.device))
            theta = torch.tensor(-torch.pi / 4, dtype=ref.dtype, device=ref.device)
            c = torch.cos(theta)
            s = torch.sin(theta)
            zero = torch.zeros((), dtype=ref.dtype, device=ref.device)
            xsgauss_rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])
            D = torch.stack([torch.stack([1 / a**2, zero]), torch.stack([zero, 1 / b**2])])
            Q = xsgauss_rot @ D @ xsgauss_rot.T
            qform = Q[0, 0] * dx**2 + 2 * Q[0, 1] * dx * dy + Q[1, 1] * dy**2
            xsgauss0 = 1 / (2. * torch.pi * a * b)
            xsgauss_arr = xsgauss0 * torch.exp(-0.5 * qform)
            xsgauss_arr = xsgauss_arr / xsgauss_arr.sum() * gq * I0
            xsringauss_arr = xsringauss_arr / xsringauss_arr.sum() * I0 * (1 - gq) + self.floor_component() + self.bg_gauss_component() + xsgauss_arr
            return imutils.gauss_blur(xsringauss_arr, self.gauss_blur_sigma, self.psize)

        cos_phi = np.cos(self.phi+np.pi/2)
        sin_phi = np.sin(self.phi+np.pi/2)
        xx, yy = np.meshgrid(self.X, self.Y, indexing='xy')
        x0 = xx * cos_phi / self.stretchx + yy * sin_phi / self.stretchy
        y0 = -yy * cos_phi / self.stretchy + xx * sin_phi / self.stretchx
        slash_axis = (x0 + y0) / np.sqrt(2)

        R1 = (x0 - self.d / np.sqrt(2))**2 + (y0 - self.d / np.sqrt(2))**2
        R2 = x0**2 + y0**2

        xsringauss_arr = np.zeros((self.dim, self.dim))
        mask = (R1 > self.Rn**2) & (R2 < self.Rp**2)
        xs0 = (2 * self.I0 * (1 - self.gq) / np.pi) / (self.Rp**2 - self.Rn**2 * (1 + self.d / self.Rp)
                                                       - (1 - self.f) * (self.d * self.Rn**2 / self.Rp))
        xsringauss_arr[mask] = xs0 * ((1 - self.f) * slash_axis[mask] / self.Rp + 1 + self.f) * 0.5

        dx = x0 - (self.d + self.Rn) / np.sqrt(2)
        dy = y0 - (self.d + self.Rn) / np.sqrt(2)
        xsgauss_rot = np.array([[np.cos(-np.pi/4), -np.sin(-np.pi/4)],
                                [np.sin(-np.pi/4),  np.cos(-np.pi/4)]])
        D = np.array([[1 / self.a**2, 0], [0, 1 / self.b**2]])
        Q = xsgauss_rot @ D @ xsgauss_rot.T
        qform = Q[0, 0] * dx**2 + 2 * Q[0, 1] * dx * dy + Q[1, 1] * dy**2
        xsgauss0 = 1 / (2. * np.pi * self.a * self.b)
        xsgauss_arr = xsgauss0 * np.exp(-0.5 * qform)
        
        xsgauss_arr = xsgauss_arr/np.sum(xsgauss_arr)*self.gq*self.I0
        xsringauss_arr = xsringauss_arr/np.sum(xsringauss_arr)*self.I0*(1-self.gq) + self.floor_component() + self.bg_gauss_component() + xsgauss_arr
        xsringauss_arr = imutils.gauss_blur(xsringauss_arr, self.gauss_blur_sigma, self.psize)
        return xsringauss_arr
    
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
            if uv.ndim != 2 or (uv.shape[0] != 2 and uv.shape[1] != 2):
                raise ValueError("uv must have shape (2, N) or (N, 2).")
            if uv.shape[0] != 2:
                uv = uv.transpose(0, 1)

            dphi = torch.pi / 4
            phi = torch.as_tensor(self.phi, dtype=uv.dtype, device=uv.device)
            cos_term = torch.cos(phi + dphi)
            sin_term = torch.sin(phi + dphi)
            u_rot = uv[0] * cos_term + uv[1] * sin_term
            v_rot = -uv[0] * sin_term + uv[1] * cos_term
            uv_rot = torch.stack([u_rot, v_rot], dim=0)
            stretch = torch.tensor([self.stretchx, self.stretchy], dtype=uv.dtype, device=uv.device)
            anaVis = vis.stretch_vis(uv_rot, lambda uv: vis.slashed_crescent(uv, (1-self.gq)*self.I0, self.Rp*1e-6/206265, self.Rn*1e-6/206265, -self.ecn, self.f), stretch)
            anaVis += vis.stretch_vis(uv_rot, lambda uv: vis.circ_gauss(uv, self.bg_gauss_flux, self.bg_gauss_sigma*1e-6/206265, offset=(0,0)), stretch)
            anaVis += vis.stretch_vis(uv_rot, lambda uv: vis.circ_gauss(uv, self.floor_brightness, self.floor_sigma*1e-6/206265, offset=(-self.d*1e-6/206265,0)), stretch)
            anaVis += vis.stretch_vis(uv_rot, lambda uv: vis.elliptical_gauss(uv, self.gq*self.I0, self.a*1e-6/206265, self.b*1e-6/206265, torch.pi/2, ((-self.d-self.Rn)*1e-6/206265,0)), stretch)
            anaVis = vis.blur_by_gauss_kernel(uv_rot, anaVis, self.gauss_blur_sigma*1e-6/206265)
        return anaVis

    def key_params(self):
        """Returns the key parameters of the model.

        Returns
        -------
        dict
            A dictionary containing the key parameters of the model.
        """
        # crescent diameter
        d_hat = self.Rp + self.Rn
        # fractional crescent width
        fw = (self.Rp - self.Rn + self.gauss_blur_sigma) / d_hat
        # orientation
        phi = self.phi * 180/np.pi
        # sharpness
        s_hat = self.gauss_blur_sigma / d_hat
        log10_s_hat = np.log10(s_hat)
        # flux ratio between emission floor and mean crescent brightness
        f_hat = self.floor_brightness * ((2*self.Rp + self.gauss_blur_sigma)**2 - (2*self.Rn - self.gauss_blur_sigma)**2) / (8 * self.floor_sigma**2 * (self.I0 + self.floor_brightness))
        log10_f_hat = np.log10(f_hat)

        out_dict = {
            'd_hat': d_hat,
            'fw': fw,
            'phi': phi,
            's_hat': s_hat,
            'log10_s_hat': log10_s_hat,
            'f_hat': f_hat,
            'log10_f_hat': log10_f_hat
        }

        return out_dict