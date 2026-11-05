from contextlib import contextmanager, redirect_stdout, redirect_stderr
from functools import lru_cache
import os
import itertools as it

import numpy as np
import torch


def is_torch(x):
    """Return whether x is a Torch tensor."""
    return torch.is_tensor(x)


def to_torch(x, like=None, complex_ok=False):
    """Convert to a tensor using like's device/dtype; leave existing tensors unchanged."""
    if torch.is_tensor(x):
        return x
    if like is not None and torch.is_tensor(like):
        if complex_ok:
            dtype = like.dtype
        else:
            dtype = like.real.dtype if like.is_complex() else like.dtype
        return torch.as_tensor(x, dtype=dtype, device=like.device)
    return torch.as_tensor(x)


def to_numpy(x):
    """Convert to NumPy, detaching tensors and moving them to the CPU."""
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def compute_cov(data):
    """Compute covariance from a (samples, features) NumPy-compatible array."""
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("Expected data to be a 2D array with shape (n_samples, n_features).")
    return np.cov(data, rowvar=False)


def compute_logcovdet(cov):
    """Return the log determinant of a square covariance with positive determinant."""
    cov = np.asarray(cov)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Expected cov to be a 2D square matrix.")
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Covariance matrix must be positive definite.")
    return logdet


def compute_invcov(cov):
    """Invert a square covariance matrix."""
    cov = np.asarray(cov)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("Expected cov to be a 2D square matrix.")
    return np.linalg.inv(cov)


def real_features(values):
    """Pack a complex vector as [real, imaginary], preserving NumPy/Torch type."""
    if torch.is_tensor(values):
        return torch.cat([values.real, values.imag]) if values.is_complex() else values
    values = np.asarray(values)
    return np.concatenate([values.real, values.imag]) if np.iscomplexobj(values) else values


def estimate_mc_noise(samples, calc_cov=True):
    """Summarise (draws, features) samples for any real-valued observable.

    Complex observables use concatenated real and imaginary features.
    """
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("Expected at least two draws with shape (N, features)")
    if np.iscomplexobj(samples):
        samples = np.concatenate([samples.real, samples.imag], axis=1)
    covariance = np.atleast_2d(np.cov(samples, rowvar=False)) if calc_cov else None
    return np.std(samples, axis=0, ddof=1), covariance


def _any_tensor(*args):
    return any(torch.is_tensor(arg) for arg in args)

def _as_tensors(*args, dtype=None, device=None):
    tensors = []
    for arg in args:
        if torch.is_tensor(arg):
            tensors.append(arg.to(dtype=dtype, device=device))
        else:
            tensors.append(torch.tensor(arg, dtype=dtype, device=device))
    return tensors

@lru_cache(maxsize=32)
def _image_axis(dim, fov):
    """Cache the fixed image axis used when constructing model components."""
    return np.linspace(-fov / 2, fov / 2, dim)


def image_axes(dim, fov):
    """Return independent coordinate arrays without rebuilding the grid spacing."""
    axis = _image_axis(dim, fov)
    return axis.copy(), axis.copy()


@contextmanager
def silence_output():
    """Temporarily silence stdout and stderr, restoring them even after errors."""
    with open(os.devnull, 'w') as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            yield

def tri_minimal_set(sites, tarr, tkey):
    """Determine a minimal set of triangles for closure calculation based on the available sites and their order in the observation.

    Parameters
    ----------
    sites : list
        The list of sites available in the current data.
    tarr : np.ndarray
        The array of times and corresponding sites from the observation.
    tkey : str
        The key in tarr that corresponds to the site identifiers.

    Returns
    -------
    tris : list of tuples
        A list of tuples, each containing three sites that form a triangle for closure calculation.
    """

    # determine ordering and reference site based on order of  self.tarr
    sites_ordered = [x for x in tarr['site'] if x in sites]
    ref = sites_ordered[0]
    sites_ordered.remove(ref)

    # Find all triangles that contain the ref
    tris = list(it.combinations(sites_ordered, 2))
    tris = [(ref, t[0], t[1]) for t in tris]

    return tris

def quad_minimal_set(sites, tarr, tkey):
    """returns a minimal set of quadrangles for closure amplitude"""

    # determine ordering and reference site based on order of  self.tarr
    sites_ordered = np.array([x for x in tarr['site'] if x in sites])
    ref = sites_ordered[0]

    # Loop over other sites >=3 and form minimal closure amplitude set
    quads = []
    for i in range(3, len(sites_ordered)):
        for j in range(1, i):
            if j == i-1:
                k = 1
            else:
                k = j+1

            # convetion is (12)(34)/(14)(23)
            quad = (ref, sites_ordered[i], sites_ordered[j], sites_ordered[k])
            quads.append(quad)

    return quads

def augment_sigma(data, sigma, noise_frac=0, static_noise_floor=0, noise_factor=1):
    """Augment the noise by adding additional noise in quadrature.

    Parameters
    ----------
    data : np.ndarray
        The observed data.
    sigma : np.ndarray
        The original noise standard deviation.

    Returns
    -------
    augmented_sigma : np.ndarray
        The augmented noise standard deviation.
    """
    augmented_sigma = np.sqrt(sigma**2 + (noise_frac * np.abs(data))**2 + static_noise_floor**2) * noise_factor
    return augmented_sigma

def calc_uvdist(uv_arr):
    """Calculate the UV distance for a given array of UV coordinates.

    Parameters
    ----------
    uv_arr : np.ndarray
        The array of UV coordinates with shape (N, 2), where N is the number of baselines.

    Returns
    -------
    uvdist : np.ndarray
        The UV distance for each baseline.
    """
    if uv_arr.ndim == 2:
        uvdist = np.sqrt(uv_arr[0]**2 + uv_arr[1]**2)
    else:
        uvdist = np.array([np.sqrt(uv_arr[j][0]**2 + uv_arr[j][1]**2) for j in range(len(uv_arr))])
        uvdist = np.sum(uvdist, axis=0)
    return uvdist
