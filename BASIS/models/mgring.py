from __future__ import division
from __future__ import print_function

from builtins import object
import numpy as np
import torch
import BASIS.modules.imutils as imutils
import BASIS.modules.vis as vis
import scipy.special as sp
import BASIS.modules.utils as utils


MODEL_PARAMS = {
    'I0': {"value": 1, "limits": (0.05, 10), "fixed": False}, # Total flux density (Jy)
    "Rp": {"value": 24, "limits": (10, 40), "fixed": False}, # Outer radius (uas)
    "phi": {"value": 180*np.pi/180, "limits": (0, 2*np.pi), "fixed": False}, # Orientation angle (0-2pi radians)
    'mrblur_sigma': {"value": 6, "limits": (5, 15), "fixed": False}, # m-ring Gaussian blur sigma (uas)
    'mrcoeff1': {"value": 0.5, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 1 (0-0.5)
    'mrcoeff1_phase': {"value": 0, "limits": (-np.pi, np.pi), "fixed": True}, # phase of m-ring coefficient 1 (radians)
    'mrcoeff2': {"value": 0.25, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 2 (0-0.5)
    'mrcoeff2_phase': {"value": 0.6, "limits": (-np.pi, np.pi), "fixed": False}, # phase of m-ring coefficient 2 (radians)
    'mrcoeff3': {"value": 0.1, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 3 (0-0.5)
    'mrcoeff3_phase': {"value": 0.5, "limits": (-np.pi, np.pi), "fixed": False}, # phase of m-ring coefficient 3 (radians)
    'mrcoeff4': {"value": 0.2, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 4 (0-0.5)
    'mrcoeff4_phase': {"value": 0.4, "limits": (-np.pi, np.pi), "fixed": False}, # phase of m-ring coefficient 4 (radians)
    'mrcoeff5': {"value": 0.05, "limits": (0, 0.5), "fixed": False}, # m-ring coefficient 5 (0-0.5)
    'mrcoeff5_phase': {"value": 0.2, "limits": (-np.pi, np.pi), "fixed": False}, # phase of m-ring coefficient 5 (radians)
    # things get a little weird with more coefficients, so we cap off at 5 coeffs
    "concGaussFlux": {"value": 0.6, "limits": (0, 0.99), "fixed": False}, # Fraction of flux in a central Gaussian (default is 0)
    "concGaussSigma": {"value": 50, "limits": (1, 100), "fixed": False}, # Sigma of the central Gaussian (default is 1)
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


class mgring(object):
    """Class for generating mG-ring models."""

    def __init__(self, I0=1, Rp=40, phi=0,
                 dim=128, fov=225,
                 mrblur_sigma=5,
                 concGaussFlux=0, concGaussSigma=50,
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
            mrcoeff1-n (float) : m-ring coefficient magnitudes
            mrcoeff1_phase-n (float) : m-ring coefficient phases (radians)
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
        coeff_indices = [
            int(key[len('mrcoeff'):])
            for key in kwargs
            if key.startswith('mrcoeff') and key[len('mrcoeff'):].isdigit()
        ]
        n_coeffs = max(coeff_indices, default=0)
        self.coeff_magnitudes = [kwargs.get(f'mrcoeff{i+1}', 0) for i in range(n_coeffs)]
        self.coeff_phases = [kwargs.get(f'mrcoeff{i+1}_phase', 0) for i in range(n_coeffs)]

        if utils._any_tensor([*self.coeff_magnitudes, *self.coeff_phases]):
            ref = next(value for value in [*self.coeff_magnitudes, *self.coeff_phases]
                       if torch.is_tensor(value))
            magnitudes = torch.stack([
                torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
                for value in self.coeff_magnitudes
            ])
            phases = torch.stack([
                torch.as_tensor(value, dtype=ref.dtype, device=ref.device)
                for value in self.coeff_phases
            ])
            self.coeffs = torch.polar(magnitudes, phases)
        else:
            self.coeffs = (np.asarray(self.coeff_magnitudes, dtype=float)
                           * np.exp(1j * np.asarray(self.coeff_phases, dtype=float)))
        self.concGaussFlux = concGaussFlux * self.I0
        self.concGaussSigma = concGaussSigma
        self.I0 = self.I0 * (1 - concGaussFlux)

        self.X = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.Y = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.psize = (self.X[1]-self.X[0])
        

    def concGauss_component(self):
        """Generates the central Gaussian component of the model.

        Returns:
            Central Gaussian component of the model
        """
        if utils._any_tensor([self.concGaussFlux, self.concGaussSigma, self.phi]):
            ref = next(v for v in [self.concGaussFlux, self.concGaussSigma, self.phi] if torch.is_tensor(v))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            bg_gauss_flux, bg_gauss_sigma = utils._as_tensors(self.concGaussFlux, self.concGaussSigma, dtype=ref.dtype, device=ref.device)
            cos_phi = torch.cos(self.phi+torch.pi/2)
            sin_phi = torch.sin(self.phi+torch.pi/2)
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
        bg_gauss_arr = np.exp(-(x0**2 + y0**2)/(2.*self.concGaussSigma**2))
        bg_gauss_arr = bg_gauss_arr/np.sum(bg_gauss_arr)*self.concGaussFlux
        return bg_gauss_arr
        

    def sky_map(self):
        """Generates the intensity map of the model
         
        Returns:
            Intensity map of the model
        """ 
        tensor_params = [self.I0, self.Rp, self.phi, self.mrblur_sigma,
                         self.concGaussFlux, self.concGaussSigma, self.stretchx,
                         self.stretchy, *self.coeff_magnitudes, *self.coeff_phases]
        if utils._any_tensor(tensor_params):
            ref = next(value for value in tensor_params if torch.is_tensor(value))
            X = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov/2, self.fov/2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing='xy')
            I0, Rp, phi, mrblur_fwhm, concGaussFlux, concGaussSigma, stretchx, stretchy = utils._as_tensors(self.I0, self.Rp, self.phi, self.mrblur_fwhm, self.concGaussFlux, self.concGaussSigma, self.stretchx, self.stretchy, dtype=ref.dtype, device=ref.device)
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
                coeff_t = torch.as_tensor(coeff, dtype=mring.dtype, device=ref.device)
                ik = _modified_bessel_in(k, arg)
                mring = mring + coeff_t * ik * torch.polar(torch.ones_like(phi0), k * phi0) + torch.conj(coeff_t) * ik * torch.polar(torch.ones_like(phi0), -k * phi0)
            mring_arr = gauss_blur * torch.real(mring)
            mring_arr = torch.real(mring_arr) / mring_arr.sum() * I0
            concGaussArr = self.concGauss_component()
            mring_arr = mring_arr + concGaussArr
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
        concGaussArr = self.concGauss_component()
        mring_arr = mring_arr + concGaussArr

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

            phi = torch.as_tensor(self.phi, dtype=uv.dtype, device=uv.device)
            stretchx = torch.as_tensor(self.stretchx, dtype=uv.dtype, device=uv.device)
            stretchy = torch.as_tensor(self.stretchy, dtype=uv.dtype, device=uv.device)
            u_rot = stretchx * uv[0] * torch.cos(phi) + stretchy * uv[1] * torch.sin(phi)
            v_rot = stretchx * uv[0] * torch.sin(phi) - stretchy * uv[1] * torch.cos(phi)
            uv_rot = torch.stack([u_rot, v_rot], dim=0)
            stretch = torch.tensor([self.stretchx, self.stretchy], dtype=uv.dtype, device=uv.device)
            rho_uv = torch.sqrt(u_rot**2 + v_rot**2)
            phi_uv = torch.atan2(v_rot, u_rot)
            x = torch.pi * self.d * 1e-6 / 206265 * rho_uv
            anaVis = _bessel_jn(0, x).to(torch.complex64)
            for m, coeff in enumerate(self.coeffs):
                k = m + 1
                coeff_t = torch.as_tensor(coeff, dtype=anaVis.dtype, device=uv.device)
                j_pos = _bessel_jn(k, x)
                e_pos = torch.polar(torch.ones_like(phi_uv), k * phi_uv)
                e_neg = torch.polar(torch.ones_like(phi_uv), -k * phi_uv)
                fourier_phase = torch.polar(
                    torch.ones_like(phi_uv),
                    torch.full_like(phi_uv, -k * torch.pi / 2),
                )
                anaVis = anaVis + fourier_phase * j_pos * (
                    coeff_t * e_pos + torch.conj(coeff_t) * e_neg
                )

            blur = torch.exp(- (torch.pi * self.mrblur_fwhm * 1e-6 / 206265 * rho_uv) ** 2 / (4 * torch.log(torch.tensor(2.0, dtype=uv.dtype, device=uv.device))))
            anaVis = anaVis * self.I0 * blur
            if self.concGaussFlux > 0:
                anaVis += vis.stretch_vis(uv, lambda uv: vis.circ_gauss(uv, self.concGaussFlux, self.concGaussSigma*1e-6/206265, offset=(0,0)), stretch=stretch)
            return anaVis
