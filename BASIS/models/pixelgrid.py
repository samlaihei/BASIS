from __future__ import division
from __future__ import print_function

from builtins import object
import re
import numpy as np
import torch
import torch.nn.functional as F
import BASIS.modules.imutils as imutils
import BASIS.modules.vis as vis
import BASIS.modules.utils as utils

DEFAULT_GRID_N = 6
BASE_PIXEL_LIMS = (0, 10)
GRID_FOV_FRAC = 0.5 # Fraction of the full image FOV occupied by the interpolated pixelgrid.

def _build_model_params(grid_n=DEFAULT_GRID_N):
    params = {
        'I0': {"value": 1, "limits": (0.05, 10), "fixed": False}, # Total flux density (Jy)
        "gauss_blur_sigma": {"value": 0, "limits": (0, 10), "fixed": True}, # Gaussian blur sigma (in uas)
        "interp_order": {"value": 3, "limits": (0, 3), "fixed": True}, # 0=nearest, 1=bilinear, >=2 bicubic
    }
    center_idx = (grid_n // 2) * grid_n + (grid_n // 2) + 1
    gamma_dist = np.random.gamma(shape=2.0, scale=1.0, size=grid_n * grid_n)
    inv_gamma_dist = 1 / np.clip(gamma_dist, a_min=1e-6, a_max=None)
    inv_gamma_dist = inv_gamma_dist / np.max(inv_gamma_dist) * (BASE_PIXEL_LIMS[1] - BASE_PIXEL_LIMS[0]) + BASE_PIXEL_LIMS[0]
    log_uniform_dist = np.exp(np.random.uniform(low=np.log(BASE_PIXEL_LIMS[0] + 1e-6), high=np.log(BASE_PIXEL_LIMS[1]), size=grid_n * grid_n))
    for idx in range(grid_n * grid_n):
        pix_key = f"pix{idx + 1}"
        if idx + 1 == center_idx:
            params[pix_key] = {"value": BASE_PIXEL_LIMS[1], "limits": BASE_PIXEL_LIMS, "fixed": True}
        else:
            params[pix_key] = {"value": log_uniform_dist[idx], "limits": BASE_PIXEL_LIMS, "fixed": False}
    return params


MODEL_PARAMS = _build_model_params()


def _interp_mode_from_order(order):
    if order <= 0.5:
        return 'nearest'
    if order <= 1.5:
        return 'bilinear'
    return 'bicubic'

def _upsample_grid(grid2d, dim, mode):
    grid4d = grid2d.unsqueeze(0).unsqueeze(0)
    if mode == 'nearest':
        return F.interpolate(grid4d, size=(dim, dim), mode=mode)[0, 0]
    return F.interpolate(grid4d, size=(dim, dim), mode=mode, align_corners=False)[0, 0]


def _coarse_canvas_dim(grid_n):
    frac = float(GRID_FOV_FRAC)
    if frac <= 0 or frac > 1:
        raise ValueError("GRID_FOV_FRAC must be in (0, 1].")
    return max(grid_n, int(np.ceil(grid_n / frac)))


def _embed_in_coarse_canvas_torch(coarse):
    canvas_dim = _coarse_canvas_dim(coarse.shape[0])
    canvas = torch.zeros((canvas_dim, canvas_dim), dtype=coarse.dtype, device=coarse.device)
    start = (canvas_dim - coarse.shape[0]) // 2
    end = start + coarse.shape[0]
    canvas[start:end, start:end] = coarse
    return canvas


def _embed_in_coarse_canvas_numpy(coarse):
    canvas_dim = _coarse_canvas_dim(coarse.shape[0])
    canvas = np.zeros((canvas_dim, canvas_dim), dtype=coarse.dtype)
    start = (canvas_dim - coarse.shape[0]) // 2
    end = start + coarse.shape[0]
    canvas[start:end, start:end] = coarse
    return canvas

class pixelgrid(object):
    """Class for generating geometric pixelgrid source models."""

    def __init__(self, I0=1,
                 dim=128, fov=225,
                 gauss_blur_sigma=0,
                 interp_order=3,
                 interp_mode=None,
                 **kwargs):
        """Creates a pixelgrid source model.

        Args:
            I0 (float) : Total flux (Jy)
            dim (int) : Dimensions of the image along an axis (square image)
            fov (int) : Field of view (in uas)
            gauss_blur_sigma (float) : Gaussian blur sigma (in uas)
            interp_order (float) : Interpolation order (0=nearest, 1=bilinear, >=2 bicubic)
            interp_mode (str) : Optional direct interpolation mode override ('nearest', 'bilinear', 'bicubic')
            kwargs (dict) : Pixel parameters as pix1..pixN2

        Return:
            Pixelgrid source model with parameters
        """
        self.MODEL_PARAMS = MODEL_PARAMS
        self.model_name = 'pixelgrid'

        self.I0 = I0
        self.gauss_blur_sigma = gauss_blur_sigma
        self.interp_order = interp_order
        self.interp_mode = interp_mode

        self.dim = dim
        self.fov = fov

        pix_params = []
        for key, value in kwargs.items():
            match = re.fullmatch(r"pix(\d+)", key)
            if match:
                pix_params.append((int(match.group(1)), value))

        pix_params.sort(key=lambda x: x[0])
        self.pixel_values = [v for _, v in pix_params]
        if len(self.pixel_values) == 0:
            self.pixel_values = [1.0] * (DEFAULT_GRID_N * DEFAULT_GRID_N)

        self.grid_n = int(np.sqrt(len(self.pixel_values)))
        if self.grid_n * self.grid_n != len(self.pixel_values):
            raise ValueError("Number of pixel parameters must be a perfect square.")

        self.X = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.Y = np.linspace(-self.fov/2, self.fov/2,self.dim)
        self.psize = (self.X[1]-self.X[0])
    
    def sky_map(self):
        """Generates the intensity map of the model
         
        Returns:
            Intensity map of the model
        """
        tensor_inputs = [self.I0, self.gauss_blur_sigma] + self.pixel_values
        if utils._any_tensor(tensor_inputs):
            ref = next(v for v in tensor_inputs if torch.is_tensor(v))
            pix_tensors = utils._as_tensors(*self.pixel_values, dtype=ref.dtype, device=ref.device)
            coarse = torch.stack(list(pix_tensors)).reshape(self.grid_n, self.grid_n)

            if self.interp_mode is not None:
                mode = self.interp_mode
            else:
                mode = _interp_mode_from_order(float(torch.as_tensor(self.interp_order).item()))
            coarse_canvas = _embed_in_coarse_canvas_torch(coarse)
            model = _upsample_grid(coarse_canvas, self.dim, mode)

            total_flux = torch.sum(model)
            model = model / torch.clamp(total_flux, min=torch.finfo(model.dtype).eps)
            I0 = utils._as_tensors(self.I0, dtype=ref.dtype, device=ref.device)[0]
            model = model * I0
            model = imutils.gauss_blur(model, self.gauss_blur_sigma, self.psize)
            model = torch.clamp(model, min=0)
            return model

        coarse = np.asarray(self.pixel_values, dtype=float).reshape(self.grid_n, self.grid_n)
        mode = self.interp_mode if self.interp_mode is not None else _interp_mode_from_order(float(self.interp_order))
        coarse_canvas = _embed_in_coarse_canvas_numpy(coarse)
        coarse_t = torch.as_tensor(coarse_canvas, dtype=torch.float64)
        model = _upsample_grid(coarse_t, self.dim, mode).detach().cpu().numpy()

        total_flux = np.sum(model)
        if total_flux <= 0:
            raise ValueError("Interpolated pixelgrid has non-positive total flux.")
        model = model / total_flux * self.I0
        model = imutils.gauss_blur(model, self.gauss_blur_sigma, self.psize)
        model[model < 0] = 0
        return model


    def sample_vis(self, uv, ttype='direct'):
        """Samples the visibilities at given (u, v) coordinates.

        Parameters
        ----------
        uv : array_like
            The (u, v) coordinates in wavelengths.
        ttype : str
            The type of sampling method ('direct' or 'DFT').

        Returns
        -------
        vis : array_like
            The sampled visibilities.
        """
        image = self.sky_map()
        dvis = vis.DFT(image.unsqueeze(0) if torch.is_tensor(image) else np.expand_dims(image, 0), uv, xfov=self.fov, yfov=self.fov)[0]
        return dvis
