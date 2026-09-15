from __future__ import division
from __future__ import print_function

from builtins import object
import numpy as np
import torch
import BASIS.modules.imutils as imutils
import BASIS.modules.vis as vis
import BASIS.modules.utils as utils

MODEL_PARAMS = {
    "I0": {"value": 1, "limits": (0.05, 10), "fixed": False},          # Total flux density (Jy)
    "Rp": {"value": 24, "limits": (10, 40), "fixed": False},            # Outer radius (uas)
    "Rn_frac": {"value": 0.75, "limits": (0.05, 0.95), "fixed": False}, # Inner radius fraction (0-1)
    "phi": {"value": 0.0, "limits": (0, 2*np.pi), "fixed": True},      # Orientation (radians; relevant with stretch)
    "gauss_blur_sigma": {"value": 4, "limits": (0, 10), "fixed": False},# Gaussian blur sigma (uas)
    "stretchx": {"value": 1.0, "limits": (0.5, 2), "fixed": True},      # Stretch term for x-axis
    "stretchy": {"value": 1.0, "limits": (0.5, 2), "fixed": True},      # Stretch term for y-axis
}


class ring(object):
    """Centered ring model: outer disk minus inner disk."""

    def __init__(self, I0=1, Rp=42, Rn_frac=0.75, phi=0,
                 stretchx=1, stretchy=1,
                 fov=225, dim=64, gauss_blur_sigma=0):
        self.MODEL_PARAMS = MODEL_PARAMS
        self.model_name = "ring"

        self.I0 = I0
        self.Rp = Rp
        self.Rn = Rn_frac * Rp
        self.phi = phi
        self.stretchx = stretchx
        self.stretchy = stretchy
        self.fov = fov
        self.dim = dim
        self.gauss_blur_sigma = gauss_blur_sigma

        self.X = np.linspace(-self.fov / 2, self.fov / 2, self.dim)
        self.Y = np.linspace(-self.fov / 2, self.fov / 2, self.dim)
        self.psize = self.X[1] - self.X[0]

    def sky_map(self):
        """Generates ring intensity map with total flux I0."""
        if utils._any_tensor([self.I0, self.Rp, self.Rn, self.phi, self.stretchx, self.stretchy, self.gauss_blur_sigma]):
            ref = next(v for v in [self.I0, self.Rp, self.Rn, self.phi, self.stretchx, self.stretchy, self.gauss_blur_sigma] if torch.is_tensor(v))
            X = torch.linspace(-self.fov / 2, self.fov / 2, self.dim, dtype=ref.dtype, device=ref.device)
            Y = torch.linspace(-self.fov / 2, self.fov / 2, self.dim, dtype=ref.dtype, device=ref.device)
            xx, yy = torch.meshgrid(X, Y, indexing="xy")
            Rp, Rn, I0, phi, stretchx, stretchy = utils._as_tensors(
                self.Rp, self.Rn, self.I0, self.phi, self.stretchx, self.stretchy,
                dtype=ref.dtype, device=ref.device
            )

            cos_phi = torch.cos(phi + torch.pi / 2)
            sin_phi = torch.sin(phi + torch.pi / 2)
            x0 = xx * cos_phi / stretchx + yy * sin_phi / stretchy
            y0 = -yy * cos_phi / stretchy + xx * sin_phi / stretchx
            R = torch.sqrt(x0**2 + y0**2)

            mask = imutils.soft_inside(R, Rp, self.psize) * imutils.soft_outside(R, Rn, self.psize)
            ring_arr = mask / mask.sum() * I0
            return imutils.gauss_blur(ring_arr, self.gauss_blur_sigma, self.psize)

        cos_phi = np.cos(self.phi + np.pi / 2)
        sin_phi = np.sin(self.phi + np.pi / 2)
        xx, yy = np.meshgrid(self.X, self.Y, indexing="xy")
        x0 = xx * cos_phi / self.stretchx + yy * sin_phi / self.stretchy
        y0 = -yy * cos_phi / self.stretchy + xx * sin_phi / self.stretchx
        R2 = x0**2 + y0**2

        ring_arr = np.zeros((self.dim, self.dim))
        mask = (R2 < self.Rp**2) & (R2 > self.Rn**2)
        ring_arr[mask] = 1.0
        ring_arr = ring_arr / np.sum(ring_arr) * self.I0
        ring_arr = imutils.gauss_blur(ring_arr, self.gauss_blur_sigma, self.psize)
        return ring_arr

    def sample_vis(self, uv, ttype="analytical"):
        """Samples visibilities at (u, v) coordinates."""
        if ttype in ("direct", "DFT", "dft"):
            image = self.sky_map()
            dvis = vis.DFT(
                image.unsqueeze(0) if torch.is_tensor(image) else np.expand_dims(image, 0),
                uv,
                xfov=self.fov,
                yfov=self.fov
            )[0]
            return dvis

        uv = torch.as_tensor(uv)
        if uv.ndim != 2 or (uv.shape[0] != 2 and uv.shape[1] != 2):
            raise ValueError("uv must have shape (2, N) or (N, 2).")
        if uv.shape[0] != 2:
            uv = uv.transpose(0, 1)

        phi = torch.as_tensor(self.phi, dtype=uv.dtype, device=uv.device)
        cos_term = torch.cos(phi + torch.pi / 2)
        sin_term = torch.sin(phi + torch.pi / 2)
        u_rot = uv[0] * cos_term + uv[1] * sin_term
        v_rot = -uv[0] * sin_term + uv[1] * cos_term
        uv_rot = torch.stack([u_rot, v_rot], dim=0)
        stretch = torch.tensor([self.stretchx, self.stretchy], dtype=uv.dtype, device=uv.device)

        # Ring = outer disk - inner disk, with constant surface brightness and total flux I0
        denom = (self.Rp**2 - self.Rn**2)
        outer_flux = self.I0 * (self.Rp**2 / denom)
        inner_flux = self.I0 * (self.Rn**2 / denom)
        uas2rad = 1e-6 / 206265

        anaVis = vis.stretch_vis(
            uv_rot,
            lambda uv_: vis.disk(uv_, outer_flux, self.Rp * uas2rad, offset=(0, 0)),
            stretch
        )
        anaVis -= vis.stretch_vis(
            uv_rot,
            lambda uv_: vis.disk(uv_, inner_flux, self.Rn * uas2rad, offset=(0, 0)),
            stretch
        )
        anaVis = vis.blur_by_gauss_kernel(uv_rot, anaVis, self.gauss_blur_sigma * uas2rad)
        return anaVis

    def key_params(self):
        """Returns key ring parameters."""
        d_hat = self.Rp + self.Rn
        fw = (self.Rp - self.Rn + self.gauss_blur_sigma) / d_hat
        phi = self.phi * 180 / np.pi
        s_hat = self.gauss_blur_sigma / d_hat
        log_s_hat = np.log10(s_hat) if s_hat > 0 else -np.inf

        return {
            "d_hat": d_hat,
            "fw": fw,
            "phi": phi,
            "s_hat": s_hat,
            "log10_s_hat": log_s_hat,
        }