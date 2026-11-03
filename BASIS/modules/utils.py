from contextlib import contextmanager, redirect_stdout, redirect_stderr
from functools import lru_cache
import os

import numpy as np
import torch

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
