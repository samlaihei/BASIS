import torch


def _split_uv(uv):
    """Return u, v tensors from tuple/list or tensor-like uv input."""
    if isinstance(uv, (tuple, list)) and len(uv) == 2:
        u = torch.as_tensor(uv[0])
        v = torch.as_tensor(uv[1], dtype=u.dtype, device=u.device)
        return u, v

    uv_t = torch.as_tensor(uv)
    if uv_t.ndim >= 1 and uv_t.shape[0] == 2:
        return uv_t[0], uv_t[1]
    if uv_t.ndim >= 1 and uv_t.shape[-1] == 2:
        return uv_t[..., 0], uv_t[..., 1]
    raise ValueError("uv must be provided as (u, v) or as an array with a length-2 axis.")


def _to_like(value, ref):
    return torch.as_tensor(value, dtype=ref.dtype, device=ref.device)


def _phase(u, v, offset, sign=1.0):
    dx = _to_like(offset[0], u)
    dy = _to_like(offset[1], u)
    angle = sign * 2 * torch.pi * (u * dx + v * dy)
    return torch.polar(torch.ones_like(angle), angle)


def _j2(x):
    j0 = torch.special.bessel_j0(x)
    j1 = torch.special.bessel_j1(x)
    safe_x = torch.where(x == 0, torch.ones_like(x), x)
    j2 = 2 * j1 / safe_x - j0
    return torch.where(x == 0, torch.zeros_like(x), j2)

def point(uv, flux, offset):
    """Generate the visibility function for a point source.

    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    flux : float
        The total flux of the point source.
    offset : tuple
        The (dx, dy) offset of the point source center in radians.

    Returns
    -------
    vis : array_like
        The visibility function evaluated at the given (u, v) coordinates.
    """
    u, v = _split_uv(uv)
    flux = _to_like(flux, u)
    phase = _phase(u, v, offset, sign=-1.0)
    vis = flux * phase
    return vis

def circ_gauss(uv, flux, sigma, offset):
    """Generate the visibility function for a circular Gaussian source.

    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    flux : float
        The total flux of the Gaussian source.
    sigma : float
        The standard deviation of the Gaussian in radians.
    offset : tuple
        The (dx, dy) offset of the Gaussian center in radians.

    Returns
    -------
    vis : array_like
        The visibility function evaluated at the given (u, v) coordinates.
    """
    u, v = _split_uv(uv)
    flux = _to_like(flux, u)
    sigma = _to_like(sigma, u)
    r_uv = torch.sqrt(u**2 + v**2)
    phase = _phase(u, v, offset, sign=-1.0)
    vis = flux * torch.exp(-2 * (torch.pi * sigma * r_uv) ** 2) * phase
    return vis

def elliptical_gauss(uv, flux, sigma_x, sigma_y, pa, offset):
    """Generate the visibility function for an elliptical Gaussian source.

    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    flux : float
        The total flux of the Gaussian source.
    sigma_x : float
        The standard deviation along the x-axis in radians.
    sigma_y : float
        The standard deviation along the y-axis in radians.
    pa : float
        The position angle of the major axis in radians.
    offset : tuple
        The (dx, dy) offset of the Gaussian center in radians.

    Returns
    -------
    vis : array_like
        The visibility function evaluated at the given (u, v) coordinates.
    """
    u, v = _split_uv(uv)
    flux = _to_like(flux, u)
    sigma_x = _to_like(sigma_x, u)
    sigma_y = _to_like(sigma_y, u)
    pa = _to_like(pa, u)
    u_rot = u * torch.cos(pa) + v * torch.sin(pa)
    v_rot = -u * torch.sin(pa) + v * torch.cos(pa)
    r_uv = torch.sqrt((sigma_x * u_rot) ** 2 + (sigma_y * v_rot) ** 2)
    phase = _phase(u, v, offset, sign=1.0)
    vis = flux * torch.exp(-2 * (torch.pi * r_uv) ** 2) * phase
    return vis

def disk(uv, flux, radius, offset):
    """Generate the visibility function for a uniform disk source.

    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    flux : float
        The total flux of the disk source.
    radius : float
        The radius of the disk in radians.
    offset : tuple
        The (dx, dy) offset of the disk center in radians.

    Returns
    -------
    vis : array_like
        The visibility function evaluated at the given (u, v) coordinates.
    """
    u, v = _split_uv(uv)
    flux = _to_like(flux, u)
    radius = _to_like(radius, u)
    r_uv = torch.sqrt(u**2 + v**2)
    phase = _phase(u, v, offset, sign=1.0)
    x = 2 * torch.pi * radius * r_uv
    ratio = torch.where(x == 0, torch.ones_like(x), 2 * torch.special.bessel_j1(x) / x)
    vis = flux * ratio * phase
    return vis

def crescent(uv, flux, R_out, R_in, eccentricity):
    """Generate the visibility function for a crescent source.

    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    flux : float
        The total flux of the crescent source.
    R_out : float
        The outer radius of the crescent in radians.
    R_in : float
        The inner radius of the crescent in radians.
    eccentricity : float
        The eccentricity of the crescent (0 <= eccentricity < 1).

    Returns
    -------
    vis : array_like
        The visibility function evaluated at the given (u, v) coordinates.
    """
    d = eccentricity * (R_out - R_in)
    flux_outer = flux / (1 - (R_in / R_out)**2)
    flux_inner = flux_outer * (R_in / R_out)**2
    vis_out = disk(uv, flux_outer, R_out, [0, 0])
    vis_in = disk(uv, flux_inner, R_in, [d, 0])
    vis = vis_out - vis_in
    return vis

def disk_deriv(u, r_uv, R):
    """Return the disk derivative.

    Parameters
    ----------
    u : array_like
        The u coordinates in wavelengths.
    r_uv : array_like
        The uv radius in wavelengths.
    R : float
        The disk radius in radians.

    Returns
    -------
    deriv : array_like
        The derivative evaluated for the given coordinates.
    """
    u = torch.as_tensor(u)
    r_uv = torch.as_tensor(r_uv, dtype=u.dtype, device=u.device)
    R = _to_like(R, u)

    x = 2 * torch.pi * R * r_uv
    j0 = torch.special.bessel_j0(x)
    j1 = torch.special.bessel_j1(x)
    j2 = _j2(x)

    safe_r = torch.where(r_uv == 0, torch.ones_like(r_uv), r_uv)
    deriv = R * u * safe_r.pow(-2) * (torch.pi * R * (j0 - j2) - j1 / safe_r)
    return torch.where(r_uv == 0, torch.zeros_like(deriv), deriv)

def slashed_disk(uv, flux, R_out, fading, offset):
    """Generate the visibility function for a slashed disk source.

    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    flux : float
        The total flux of the slashed disk source.
    R_out : float
        The outer radius of the disk in radians.
    fading : float
        The fading parameter (0 <= fading <= 1).
    offset : tuple
        The (dx, dy) offset of the disk center in radians.
    Returns
    -------
    vis : array_like
        The visibility function evaluated at the given (u, v) coordinates.
    """
    u, v = _split_uv(uv)
    flux = _to_like(flux, u)
    R_out = _to_like(R_out, u)
    f = _to_like(fading, u)
    r_uv = torch.sqrt(u**2 + v**2)
    phase = _phase(u, v, offset, sign=1.0)

    x_out = 2 * torch.pi * R_out * r_uv
    safe_r = torch.where(r_uv == 0, torch.ones_like(r_uv), r_uv)
    disk_vis = R_out * torch.special.bessel_j1(x_out) / safe_r
    disk_vis = torch.where(r_uv == 0, torch.pi * R_out**2, disk_vis)

    ddisk = disk_deriv(u, r_uv, R_out)
    h = 2 * flux / (torch.pi * R_out**2 * (1 + f))
    h0 = f * h
    xi = (h+h0) / 2
    eta = (h-h0) / 2

    vis = xi * disk_vis + 1j * eta / (2 * torch.pi) * ddisk / R_out
    vis = vis * phase

    return vis


def slashed_crescent(uv, flux, R_out, R_in, eccentricity, fading):
    """Generate the visibility function for a slashed crescent source.

    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    flux : float
        The total flux of the slashed crescent source.
    R_out : float
        The outer radius of the crescent in radians.
    R_in : float
        The inner radius of the crescent in radians.
    eccentricity : float
        The eccentricity of the crescent (0 <= eccentricity < 1).
    fading : float
        The fading parameter (0 <= fading <= 1).

    Returns
    -------
    vis : array_like
        The visibility function evaluated at the given (u, v) coordinates.
    """
    u, v = _split_uv(uv)
    flux = _to_like(flux, u)
    R_out = _to_like(R_out, u)
    R_in = _to_like(R_in, u)
    eccentricity = _to_like(eccentricity, u)
    f = _to_like(fading, u)

    d = eccentricity * (R_out - R_in)
    r_uv = torch.sqrt(u**2 + v**2)
    h = (2 * flux / torch.pi) * (
        (R_out**2 - R_in**2 * (1 + d / R_out)) * f
        + (R_out**2 - R_in**2 * (1 - d / R_out))
    ).pow(-1)
    h0 = f * h
    xi = (h+h0)/2
    eta = (h-h0)/2

    x_out = 2 * torch.pi * R_out * r_uv
    x_in = 2 * torch.pi * R_in * r_uv
    safe_r = torch.where(r_uv == 0, torch.ones_like(r_uv), r_uv)
    disk_out = R_out * torch.special.bessel_j1(x_out) / safe_r
    disk_in = R_in * torch.special.bessel_j1(x_in) / safe_r
    disk_out = torch.where(r_uv == 0, torch.pi * R_out**2, disk_out)
    disk_in = torch.where(r_uv == 0, torch.pi * R_in**2, disk_in)


    ddisk_out = disk_deriv(u, r_uv, R_out) 
    ddisk_in = disk_deriv(u, r_uv, R_in) 

    vis_out = xi * disk_out + 1j * eta / (2 * torch.pi) * ddisk_out / R_out
    vis_in = (xi - eta * d / R_out) * disk_in + 1j * eta / (2 * torch.pi) * ddisk_in / R_out
    vis_r = vis_out - torch.polar(torch.ones_like(u), 2 * torch.pi * u * d) * vis_in


    return vis_r

def blur_by_gauss_kernel(uv, vis, sigma):
    """Blur visibilities by a Gaussian kernel in the image plane.
    
    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    vis : array_like
        The associated visibilities to be blurred.
    sigma : float
        The standard deviation of the Gaussian kernel in radians.

    Returns
    -------
    vis_blurred : array_like
        The blurred visibilities.
    """
    if sigma <= 0:
        return vis
    u, v = _split_uv(uv)
    vis = torch.as_tensor(vis, device=u.device)
    sigma = _to_like(sigma, u)
    r_uv = torch.sqrt(u**2 + v**2)
    kernel = torch.exp(-2 * (torch.pi * sigma * r_uv) ** 2)
    vis_blurred = vis * kernel
    return vis_blurred

def stretch_vis(uv, vis_fun, stretch):
    """Stretch visibilities by a factor in the image plane.
    
    Parameters
    ----------
    uv : array_like
        The (u, v) coordinates in wavelengths.
    vis_fun : callable
        A function that computes visibilities given (u, v) coordinates.
    stretch : tuple
        The stretch factors (stretch_a, stretch_b) along the x and y axes in the image plane.

    Returns
    -------
    vis_stretched : array_like
        The stretched visibilities.
    """
    u, v = _split_uv(uv)
    vis = torch.as_tensor(vis_fun(uv), device=u.device)
    stretch_a, stretch_b = stretch
    u_stretched = u * stretch_a
    v_stretched = v * stretch_b
    uv_stretched = torch.stack((u_stretched, v_stretched), dim=-1)
    vis_stretched = vis_fun(uv_stretched)
    return vis_stretched

def DFT(data, uv, xfov=225, yfov=225):
    """Compute the Discrete Fourier Transform (DFT) of an image at given (u, v) coordinates.
    
    Parameters
    ----------
    data : array_like
        The input image data (2D array) or a stack of images (3D array).
    uv : array_like
        The (u, v) coordinates in wavelengths where the DFT is evaluated.
    xfov : float
        The field of view in the x-direction in microarcseconds.
    yfov : float
        The field of view in the y-direction in microarcseconds.
        
    Returns
    -------
    vis : array_like
        The computed visibilities at the given (u, v) coordinates.
    """
    data_t = torch.as_tensor(data)
    uv_t = torch.as_tensor(uv, device=data_t.device)

    if uv_t.ndim != 2:
        raise ValueError("uv must be a 2D tensor with shape (N, 2) or (2, N).")
    if uv_t.shape[-1] == 2:
        pass
    elif uv_t.shape[0] == 2:
        uv_t = uv_t.transpose(0, 1)
    else:
        raise ValueError("uv must have one axis of length 2.")

    real_dtype = torch.promote_types(data_t.dtype, uv_t.dtype)
    if not data_t.is_floating_point():
        real_dtype = torch.float32
    data_t = data_t.to(dtype=real_dtype)
    uv_t = uv_t.to(dtype=real_dtype)

    input_ndim = data_t.ndim
    if input_ndim < 2:
        raise ValueError("data must be at least 2D.")

    if input_ndim == 2:
        leading_shape = ()
        data_t = data_t.unsqueeze(0)
    else:
        leading_shape = tuple(data_t.shape[:-2])
        data_t = data_t.reshape((-1,) + tuple(data_t.shape[-2:]))

    out_shape = leading_shape + (uv_t.shape[0],)

    ny, nx = data_t.shape[-2:]
    dx = xfov * 4.84813681109536e-12 / nx
    dy = yfov * 4.84813681109536e-12 / ny

    angx = (torch.arange(nx, device=data_t.device, dtype=real_dtype) - nx // 2) * dx + (dx / 2)
    angy = (torch.arange(ny, device=data_t.device, dtype=real_dtype) - ny // 2) * dy + (dy / 2)
    lvect = torch.sin(angx)
    mvect = torch.sin(angy)
    l, m = torch.meshgrid(lvect, mvect, indexing="xy")
    lm = torch.stack([l.reshape(-1), m.reshape(-1)], dim=0)

    imgvect = data_t.reshape((data_t.shape[0], -1))
    x = -2 * torch.pi * (uv_t @ lm).unsqueeze(0)
    visr = torch.sum(imgvect[:, None, :] * torch.cos(x), dim=-1)
    visi = torch.sum(imgvect[:, None, :] * torch.sin(x), dim=-1)

    vis = torch.complex(visr, visi).reshape(out_shape)
    if input_ndim == 2:
        vis = vis.reshape(-1)
    return vis