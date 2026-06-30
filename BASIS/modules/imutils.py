import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import fftconvolve
import cv2

def _torch_ref(*values):
    for value in values:
        if torch.is_tensor(value):
            return value
    return torch.tensor(0.0)

def _edge_width(psize, edge_width=None, edge_width_pixels=1.0, like=None):
    """Return a small positive edge width in image units for soft masks."""
    ref = like if like is not None else _torch_ref(psize, edge_width)
    psize_t = torch.as_tensor(psize, dtype=ref.dtype, device=ref.device)
    if edge_width is None:
        width = edge_width_pixels * psize_t
    else:
        width = torch.as_tensor(edge_width, dtype=ref.dtype, device=ref.device)
    eps = torch.finfo(width.dtype).eps
    return torch.clamp(width, min=eps)


def soft_inside(radius, boundary, psize, edge_width=None, edge_width_pixels=1.0):
    """Soft indicator for radius <= boundary using a sigmoid edge."""
    ref = _torch_ref(radius, boundary, psize, edge_width)
    radius_t = torch.as_tensor(radius, dtype=ref.dtype, device=ref.device)
    boundary_t = torch.as_tensor(boundary, dtype=ref.dtype, device=ref.device)
    width = _edge_width(psize, edge_width=edge_width, edge_width_pixels=edge_width_pixels, like=radius_t)
    return torch.sigmoid((boundary_t - radius_t) / width)


def soft_outside(radius, boundary, psize, edge_width=None, edge_width_pixels=1.0):
    """Soft indicator for radius >= boundary using a sigmoid edge."""
    ref = _torch_ref(radius, boundary, psize, edge_width)
    radius_t = torch.as_tensor(radius, dtype=ref.dtype, device=ref.device)
    boundary_t = torch.as_tensor(boundary, dtype=ref.dtype, device=ref.device)
    width = _edge_width(psize, edge_width=edge_width, edge_width_pixels=edge_width_pixels, like=radius_t)
    return torch.sigmoid((radius_t - boundary_t) / width)

def centroid(imgarr):
    """Compute the centroid of an image array.
    
    Parameters
    ----------
    imgarr : np.ndarray
        2D image array to compute the centroid of.

    Returns
    -------
    tuple
        (x, y) coordinates of the centroid.
    """
    total_flux = np.sum(imgarr)
    x_coords = np.arange(imgarr.shape[1])
    y_coords = np.arange(imgarr.shape[0])
    x_centroid = np.sum(np.sum(imgarr, axis=0) * x_coords) / total_flux
    y_centroid = np.sum(np.sum(imgarr, axis=1) * y_coords) / total_flux
    return (x_centroid, y_centroid)

def shift_fft(imgarr, shift):
    """Shift an image array by a given amount using Fourier shift theorem.
    
    Parameters
    ----------
    imgarr : np.ndarray
        2D image array to shift.
    shift : tuple
        (x, y) shift in pixels.

    Returns
    -------
    np.ndarray
        Shifted image array.
    """
    shifted_img = cv2.warpAffine(imgarr, np.float32([[1, 0, shift[0]], [0, 1, shift[1]]]), (imgarr.shape[1], imgarr.shape[0]), flags=cv2.INTER_CUBIC)
    return shifted_img

def gauss_blur(image, sigma, psize):
    """Applies Gaussian blur to the image
    
    Args:
        image (2D array) : Input image to be blurred
        sigma (float) : Standard deviation of the Gaussian kernel in physical units
        psize (float) : Pixel size in physical units
        
    Returns:
        Blurred image
    """
    if torch.is_tensor(image) or torch.is_tensor(sigma) or torch.is_tensor(psize):
        image_t = image if torch.is_tensor(image) else torch.as_tensor(image)
        sigma_t = torch.as_tensor(sigma, dtype=image_t.dtype, device=image_t.device)
        psize_t = torch.as_tensor(psize, dtype=image_t.dtype, device=image_t.device)
        if float(sigma_t.detach().cpu()) <= 0:
            return image_t

        kernel_size = int(6 * float((sigma_t / psize_t).detach().cpu()))
        if kernel_size % 2 == 0:
            kernel_size += 1
        ax = torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size, dtype=image_t.dtype, device=image_t.device)
        xx, yy = torch.meshgrid(ax, ax, indexing='xy')
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * (sigma_t / psize_t)**2))
        kernel = kernel / kernel.sum()
        blurred_image = F.conv2d(image_t.unsqueeze(0).unsqueeze(0), kernel.view(1, 1, kernel_size, kernel_size), padding=kernel_size // 2)
        return blurred_image.squeeze(0).squeeze(0)

    if sigma <= 0:
        return image
    kernel_size = int(6 * sigma / psize)
    if kernel_size % 2 == 0:
        kernel_size += 1
    ax = np.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * (sigma / psize)**2))
    kernel /= np.sum(kernel)
    blurred_image = fftconvolve(image, kernel, mode='same')
    return blurred_image

