from __future__ import division
from __future__ import print_function

from builtins import object
import numpy as np
import torch
import GCFit.modules.imutils as imutils
import GCFit.modules.vis as vis
import scipy.special as sp
import GCFit.modules.utils as utils


MODEL_PARAMS = {
    'I0': {"value": 1, "limits": (0.05, 10), "fixed": False}, # Total flux density (Jy)
    "Rp": {"value": 24, "limits": (10, 40), "fixed": False}, # Outer radius (uas)
    "phi": {"value": 180*np.pi/180, "limits": (0, 2*np.pi), "fixed": False}, # Orientation angle (0-2pi radians)
    'mrblur_sigma': {"value": 6, "limits": (5, 15), "fixed": False}, # m-ring Gaussian blur sigma (uas)
    'mrcoeff1': {"value": 0.5, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 1 (0-0.5)
    'mrcoeff2': {"value": 0.25, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 2 (0-0.5)
    'mrcoeff3': {"value": 0.1, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 3 (0-0.5)
    'mrcoeff4': {"value": 0.2, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 4 (0-0.5)
    'mrcoeff5': {"value": 0.05, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 5 (0-0.5)
    # things get a little weird with more coefficients, so we cap off at 5 coeffs
    'stretchx': {"value": 1, "limits": (0.5, 2), "fixed": True}, # Stretch term for the x-axis (default is 1)
    'stretchy': {"value": 1, "limits": (0.5, 2), "fixed": True} # Stretch term for the y-axis (default is 1)
}

def _bessel_jn(n, x):
    """Compute J_n(x) for small non-negative integer n using recurrence."""
    if n == 0:
        return torch.special.bessel_j0(x)
    if n == 1:
        return torch.special.bessel_j1(x)

    j_nm1 = torch.special.bessel_j0(x)
    j_n = torch.special.bessel_j1(x)
    safe_x = torch.where(x == 0, torch.ones_like(x), x)
    for k in range(1, n):
        j_np1 = (2.0 * k / safe_x) * j_n - j_nm1
        if (k + 1) % 2 == 1:
            j_np1 = torch.where(x == 0, torch.zeros_like(j_np1), j_np1)
        j_nm1, j_n = j_n, j_np1
    return j_n


def _modified_bessel_in(n, x):
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


class mring(object):
    """Class for generating m-ring models."""

    def __init__(self, I0=1, Rp=40, phi=0,
                 dim=128, fov=225,
                 mrblur_sigma=5,
                 stretchx=1, stretchy=1,
                 **kwargs):
        """Creates an m-ring model.

        Args:
            I0 (float) : Total flux (Jy)
            Rp (float) : Ring radius (in uas)
            phi (float) : Position angle (0-pi radians)
            dim (int) : Dimensions of the image along an axis (square image)
            fov (int) : Field of view (in uas)
            mrblur_sigma (float) : m-ring Gaussian blur sigma (in uas)
            stretchx (float) : Stretch term for the x-axis (default is 1)
            stretchy (float) : Stretch term for the y-axis (default is 1)
            mrcoeff1-n (complex) : m-ring coefficients
        Return:
            m-ring model with parameters
        """
        self.MODEL_PARAMS = MODEL_PARAMS
        self.I0 = I0
        self.Rp = Rp
        self.d = Rp * 2
        self.phi = phi
        self.dim = dim
        self.fov = fov
        self.mrblur_sigma = mrblur_sigma
        self.mrblur_fwhm = 2.355 * self.mrblur_sigma
        self.stretchx = stretchx
        self.stretchy = stretchy
        n_coeffs = sum(1 for key in kwargs if key.startswith('mrcoeff'))
        self.coeffs = [kwargs.get(f'mrcoeff{i+1}', 0) for i in range(n_coeffs)]
        

        self.X = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.Y = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.psize = (self.X[1]-self.X[0])
        

    
    def sky_map(self):
        """Generates the intensity map of the model
         
        Returns:
            Intensity map of the model
        """ 
        if utils._any_tensor([self.I0, self.Rp, self.phi, self.mrblur_sigma, self.stretchx, self.stretchy, *self.coeffs]):
            ref = next(v for v in [self.I0, self.Rp, self.phi, self.mrblur_sigma, self.stretchx, self.stretchy, *self.coeffs] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            I0, Rp, phi, mrblur_fwhm, stretchx, stretchy = utils._as_tensors(self.I0, self.Rp, self.phi, self.mrblur_fwhm, self.stretchx, self.stretchy, dtype=ref.dtype, device=ref.device)
            cos_phi = torch.cos(phi)
            sin_phi = torch.sin(phi)
            x0 = xx * cos_phi / stretchx + yy * sin_phi / stretchy
            y0 = -yy * cos_phi / stretchy + xx * sin_phi / stretchx
            r0 = torch.sqrt(x0**2 + y0**2)
            phi0 = torch.atan2(y0, x0)
            arg = 4 * torch.log(torch.tensor(2.0, dtype=ref.dtype, device=ref.device)) * r0 * self.d / mrblur_fwhm**2
            gauss_blur = 4 * torch.log(torch.tensor(2.0, dtype=ref.dtype, device=ref.device)) * I0 / (mrblur_fwhm**2)
            gauss_blur = gauss_blur * torch.exp(-4 * torch.log(torch.tensor(2.0, dtype=ref.dtype, device=ref.device)) * (r0**2 + Rp**2) / mrblur_fwhm**2)
            mring = _modified_bessel_in(0, arg).to(torch.complex64)
            for m, coeff in enumerate(self.coeffs):
                k = m + 1
                coeff_t = torch.as_tensor(coeff, dtype=ref.dtype, device=ref.device)
                ik = _modified_bessel_in(k, arg)
                mring = mring + coeff_t * ik * torch.polar(torch.ones_like(phi0), k * phi0) + torch.conj(coeff_t) * ik * torch.polar(torch.ones_like(phi0), -k * phi0)
            mring_arr = gauss_blur * torch.real(mring)
            mring_arr = torch.real(mring_arr) / mring_arr.sum() * I0
            return mring_arr

        A = 4 * np.log(2) * self.I0 / (self.mrblur_fwhm**2)
        cos_phi = np.cos(self.phi)
        sin_phi = np.sin(self.phi)
        xx, yy = np.meshgrid(self.X, self.Y, indexing='xy')
        x0 = xx * cos_phi / self.stretchx + yy * sin_phi / self.stretchy
        y0 = -yy * cos_phi / self.stretchy + xx * sin_phi / self.stretchx
        r0 = np.sqrt(x0**2 + y0**2)
        phi0 = np.arctan2(y0, x0)

        arg = 4 * np.log(2) * r0 * self.d / self.mrblur_fwhm**2
        gauss_blur = A * np.exp(-4 * np.log(2) * (r0**2 + self.Rp**2) / self.mrblur_fwhm**2)

        mring = sp.iv(0, arg) + 0j
        for m, coeff in enumerate(self.coeffs):
            k = m + 1
            mring += coeff * sp.iv(k, arg) * np.exp(1j * k * phi0)
            mring += np.conj(coeff) * sp.iv(-k, arg) * np.exp(-1j * k * phi0)

        mring_arr = gauss_blur * np.real(mring)

        mring_arr = np.real(mring_arr)/np.sum(mring_arr)*self.I0

        return mring_arr

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

            dphi = torch.pi / 2
            phi = torch.as_tensor(self.phi, dtype=uv.dtype, device=uv.device)
            cos_term = torch.cos(phi + dphi)
            sin_term = torch.sin(phi + dphi)
            stretchx = torch.as_tensor(self.stretchx, dtype=uv.dtype, device=uv.device)
            stretchy = torch.as_tensor(self.stretchy, dtype=uv.dtype, device=uv.device)
            u_rot = uv[0] * cos_term * stretchx + uv[1] * sin_term * stretchy
            v_rot = -uv[0] * sin_term * stretchx + uv[1] * cos_term * stretchy
            rho_uv = torch.sqrt(u_rot**2 + v_rot**2)
            phi_uv = torch.atan2(v_rot, u_rot)
            x = torch.pi * self.d * 1e-6 / 206265 * rho_uv

            anaVis = _bessel_jn(0, x).to(torch.complex64)
            for m, coeff in enumerate(self.coeffs):
                k = m + 1
                coeff_t = torch.as_tensor(coeff, dtype=uv.dtype, device=uv.device)
                j_pos = _bessel_jn(k, x)
                j_neg = ((-1) ** k) * j_pos
                e_pos = torch.polar(torch.ones_like(phi_uv), k * phi_uv)
                e_neg = torch.polar(torch.ones_like(phi_uv), -k * phi_uv)
                anaVis = anaVis + coeff_t * j_pos * e_pos + coeff_t * j_neg * e_neg

            blur = torch.exp(- (torch.pi * self.mrblur_fwhm * 1e-6 / 206265 * rho_uv) ** 2 / (4 * torch.log(torch.tensor(2.0, dtype=uv.dtype, device=uv.device))))
            anaVis = anaVis * self.I0 * blur
            return anaVis