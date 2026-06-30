from __future__ import division
from __future__ import print_function

from builtins import object
import re
import numpy as np
import torch
import json

from BASIS.models import (
    point,
    disk,
    gauss,
    xsring,
    xsringauss,
    sdisk,
    mring,
    pixelgrid,
)
from BASIS.modules import vis


def _batch_image(image):
    if torch.is_tensor(image):
        return image.unsqueeze(0)
    return np.expand_dims(image, 0)

MODEL_LIST = {'point': point.point, 'disk':disk.disk, 'gauss':gauss.gauss, 'sdisk':sdisk.sdisk,
              'xsring':xsring.xsring, 'xsringauss':xsringauss.xsringauss,
            'mring': mring.mring, 'pixelgrid': pixelgrid.pixelgrid}
MODEL_PARAMS = {'point': point.MODEL_PARAMS, 'disk': disk.MODEL_PARAMS, 'gauss': gauss.MODEL_PARAMS, 'sdisk': sdisk.MODEL_PARAMS,
                   'xsring': xsring.MODEL_PARAMS, 'xsringauss': xsringauss.MODEL_PARAMS, 
                'mring': mring.MODEL_PARAMS, 'pixelgrid': pixelgrid.MODEL_PARAMS}


def _resolve_model_name(model_name):
    """Resolve model aliases such as mringN or pixelgridN to a concrete model key."""
    if model_name in MODEL_LIST:
        return model_name

    match = re.fullmatch(r"mring(\d+)", model_name)
    if match:
        ncoeff = int(match.group(1))
        if ncoeff < 1:
            raise ValueError("mring coefficient count must be >= 1.")
        return 'mring'

    match = re.fullmatch(r"pixelgrid(\d+)", model_name)
    if match:
        ngrid = int(match.group(1))
        if ngrid < 1:
            raise ValueError("pixelgrid dimension must be >= 1.")
        return 'pixelgrid'

    raise ValueError(f"Model {model_name} is not recognized.")


def _params_for_model(model_name):
    """Return required parameters for a model, including dynamic mringN coefficients."""
    resolved = _resolve_model_name(model_name)

    if resolved not in ['mring', 'pixelgrid']:
        return MODEL_PARAMS[resolved].keys()

    if resolved == 'mring':
        match = re.fullmatch(r"mring(\d+)", model_name)
        if match:
            ncoeff = int(match.group(1))
        else:
            # Keep legacy behaviour for plain "mring".
            ncoeff = len([p for p in MODEL_PARAMS['mring'].keys() if p.startswith('mrcoeff')])
            if ncoeff < 1:
                ncoeff = 1

        base_params = [p for p in MODEL_PARAMS['mring'].keys() if not p.startswith('mrcoeff')]
        coeff_params = [f"mrcoeff{i+1}" for i in range(ncoeff)]
        return base_params + coeff_params

    match = re.fullmatch(r"pixelgrid(\d+)", model_name)
    if match:
        ngrid = int(match.group(1))
    else:
        # Keep legacy behaviour for plain "pixelgrid".
        npix = len([p for p in MODEL_PARAMS['pixelgrid'].keys() if p.startswith('pix')])
        ngrid = int(np.sqrt(npix))
        if ngrid < 1:
            ngrid = 1

    base_params = [p for p in MODEL_PARAMS['pixelgrid'].keys() if not p.startswith('pix')]
    coeff_params = [f"pix{i+1}" for i in range(ngrid * ngrid)]
    return base_params + coeff_params


def _defaults_for_param(model_name, param):
    """Return defaults entry, using shared defaults for dynamic mrcoeffN parameters."""
    resolved = _resolve_model_name(model_name)
    
    if resolved not in ['mring', 'pixelgrid']:
        return MODEL_PARAMS[resolved][param]

    if resolved == 'mring':
        return MODEL_PARAMS['mring'][param]

    if param in MODEL_PARAMS['pixelgrid']:
        return MODEL_PARAMS['pixelgrid'][param]
    if re.fullmatch(r"pix\d+", param):
        return MODEL_PARAMS['pixelgrid']['pix1']
    raise KeyError(f"No defaults configured for parameter {param}.")
    
    # raise KeyError(f"No defaults configured for parameter {param}.")

class BaseModel(object):
    """Base class for all geometric models."""

    def __init__(self, params=None, model_list=['gauss'], dim=128, fov=225, 
                 randomise_params=False, random_seed=None, verbose=False):
        """Initializes the base model.

        Args:
            params (dict) : Dictionary of parameters for the model
            model_list (list) : List of models to include in the base model
            dim (int) : Dimensions of the image along an axis (square image)
            fov (int) : Field of view (in uas)
            randomise_params (bool or list) : Whether to randomize parameters within their limits. 
                If a list is provided, it should be the same length as model_list and specify whether to randomize each model's parameters.
            random_seed (int) : Random seed for reproducibility when randomizing parameters
            verbose (bool) : Whether to print verbose output
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        self.model_list = model_list
        self.dim = dim
        self.fov = fov
        self.params = {}
        self.param_limits = {}
        self.param_fixed = {}
        self.verbose = verbose

        if isinstance(randomise_params, bool):
            randomise_params = [randomise_params] * len(model_list)
        elif len(randomise_params) != len(model_list):
            randomise_params = [randomise_params[0]] * len(model_list)

        self.X = np.linspace(-self.fov/2, self.fov/2, self.dim)
        self.Y = np.linspace(-self.fov/2, self.fov/2, self.dim)

        for ind, model_name in enumerate(self.model_list):
            required_params = _params_for_model(model_name)
            for param in required_params:
                if randomise_params[ind]:
                    low, high = _defaults_for_param(model_name, param)['limits']
                    fixed = _defaults_for_param(model_name, param)['fixed']
                    if not fixed:
                        self.params[f"{ind}_{param}"] = np.random.uniform(low, high)
                    else:
                        self.params[f"{ind}_{param}"] = _defaults_for_param(model_name, param)['value']
                else:
                    self.params[f"{ind}_{param}"] = _defaults_for_param(model_name, param)['value']
                self.param_limits[f"{ind}_{param}"] = _defaults_for_param(model_name, param)['limits']
                self.param_fixed[f"{ind}_{param}"] = _defaults_for_param(model_name, param)['fixed']
        
        if params is not None:
            # check if all required params are there, more params can be provided
            if set(self.params) not in [set(params.keys()), set(params.keys()).intersection(set(self.params))]:
                if self.verbose:
                    print("Provided parameters do not match required parameters for the model list.")
                    missing_params = set(self.params) - set(params.keys())
                    extra_params = set(params.keys()) - set(self.params)
                    print(f"Missing parameters: {missing_params}")
                    print(f"Extra parameters: {extra_params}")
                    print()

            else:
                # set self.params equal to minimal set of provided params
                self.params = {key: value for key, value in params.items() if key in self.params}

            for key in self.params:
                if key in params:
                    self.params[key] = params[key]

        # if all models are the same model, we can apply some heuristics to center the model and fix the brightest component
        if len(set(model_list)) == 1 and len(model_list) > 1:
            self.center_and_fix_brightest(center=True, fix_brightest=True)
            if self.verbose:
                print("All models are the same, centering on brightest component and fixing its flux.")

    def load_params_from_file(self, filepath):
        """Reads parameter properties from a json file and updates the model's parameters, limits, and fixed status.

        Args:
            filepath (str) : Path to the parameter properties file
        """
        with open(filepath, 'r') as f:
            data = json.load(f)
            if self.verbose:
                print(f"Loaded parameter properties from {filepath}.")
        for key in self.params.keys():
            if key in data:
                self.params[key] = data[key]['value']
                self.param_limits[key] = (data[key]['limit_low'], data[key]['limit_high'])
                self.param_fixed[key] = data[key]['fixed']
                if self.verbose:
                    print(f"Updated parameter {key}: value={self.params[key]}, limits={self.param_limits[key]}, fixed={self.param_fixed[key]}")
            else:
                if self.verbose:
                    print(f"Parameter {key} not found in file, keeping existing value and limits.")

        return

    def save_params_to_file(self, filepath):
        """Saves the model's parameters, limits, and fixed status to a json file.

        Args:
            filepath (str) : Path to save the parameter properties file
        """
        data = {}
        for key in self.params.keys():
            data[key] = {
                'value': self.params[key],
                'limit_low': self.param_limits[key][0],
                'limit_high': self.param_limits[key][1],
                'fixed': self.param_fixed[key]
            }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved parameter properties to {filepath}.")
        return

    def center_and_fix_brightest(self, center=True, fix_brightest=True, brightest_value=None):
        """Centers the model on the brightest point and optionally fixes the parameters of the brightest component.

        Args:
            center (bool) : Whether to center the model on the brightest point
            fix_brightest (bool) : Whether to fix the parameters of the brightest component after centering
        """
        if not center and not fix_brightest:
            return

        # Consider the first model as the brightest
        if fix_brightest:
            if brightest_value is not None:
                self.params['0_I0'] = brightest_value
            else:
                self.params['0_I0'] = self.param_limits['0_I0'][1]  # Set to max limit value
        self.param_fixed['0_I0'] = fix_brightest  # Fix I0

        if center and '0_xoff' in self.params and '0_yoff' in self.params:
            self.params['0_xoff'] = 0  # Center x offset
            self.params['0_yoff'] = 0  # Center y offset
            self.param_fixed['0_xoff'] = center  # Fix x offset
            self.param_fixed['0_yoff'] = center  # Fix y offset
        
        return


    def randomise_params(self, random_seed=None):
        """Randomizes parameters within their limits.

        Args:
            random_seed (int) : Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        for key in self.params.keys():
            if not self.param_fixed[key]:
                low, high = self.param_limits[key]
                self.params[key] = np.random.uniform(low, high)

    def num_params(self):
        """Returns the number of parameters in the model."""
        return len(self.params)

    def set_params(self, new_params, normalised=False):
        """Sets new parameters for the model from a vector.

        Args:
            new_params (list or array) : List or array of new parameter values in the same order as self.params.keys()
            normalised (bool) : Whether the new parameters are normalized (between 0 and 1) or not
        """
        if len(new_params) != len(self.params):
            raise ValueError(f"Expected {len(self.params)} parameters, got {len(new_params)}.")
        for i, key in enumerate(self.params.keys()):
            low, high = self.param_limits[key]
            value = new_params[i]
            if normalised:
                value = low + value * (high - low)

            if torch.is_tensor(value):
                if torch.any(value < low) or torch.any(value > high):
                    raise ValueError(f"Parameter {key} value {value} is out of bounds ({low}, {high}).")
            else:
                if not (low <= value <= high):
                    raise ValueError(f"Parameter {key} value {value} is out of bounds ({low}, {high}).")

            self.params[key] = value
        

    def sky_map(self, params=None):
        """Generates the combined sky map from all models.

        Returns:
            Combined sky map (2D array)
        """
        image = None
        if params is not None:
            self.set_params(params)
        for ind, model_name in enumerate(self.model_list):
            model_params = {key.split('_', 1)[1]: value for key, value in self.params.items() if key.startswith(f"{ind}_")}
            required_params = set(_params_for_model(model_name))
            if set(model_params.keys()) != required_params:
                raise ValueError(f"Parameters for model {model_name} do not match required parameters.")                    
            model_key = _resolve_model_name(model_name)
            model_instance = MODEL_LIST[model_key](dim=self.dim, fov=self.fov, **model_params)
            model_image = model_instance.sky_map()
            if image is None:
                image = torch.zeros_like(model_image) if torch.is_tensor(model_image) else np.zeros_like(model_image)
            if torch.is_tensor(image) or torch.is_tensor(model_image):
                if not torch.is_tensor(image):
                    image = torch.as_tensor(image, dtype=model_image.dtype, device=model_image.device)
                if not torch.is_tensor(model_image):
                    model_image = torch.as_tensor(model_image, dtype=image.dtype, device=image.device)
            image = image + model_image
        return image

    def sample_vis(self, uv, ttype='analytical'):
        """Samples the visibility data from the combined sky map.

        Args:
            uv (2D array) : UV coordinates for sampling
        Returns:
            Sampled visibility data (1D array)
        """
        uv_t = torch.as_tensor(uv)
        if uv_t.ndim != 2 or (uv_t.shape[0] != 2 and uv_t.shape[1] != 2):
            raise ValueError("uv must have shape (2, N) or (N, 2).")

        if uv_t.shape[0] == 2:
            nvis = uv_t.shape[1]
        else:
            nvis = uv_t.shape[0]

        if ttype == 'direct' or ttype=='DFT' or ttype=='dft':
            dvis = vis.DFT(_batch_image(self.sky_map()), uv_t, xfov=self.fov, yfov=self.fov)[0]
            return dvis
        
        else:
            avis = torch.zeros(nvis, dtype=torch.complex64, device=uv_t.device)
            for ind, model_name in enumerate(self.model_list):
                model_params = {key.split('_', 1)[1]: value for key, value in self.params.items() if key.startswith(f"{ind}_")}
                required_params = set(_params_for_model(model_name))
                if set(model_params.keys()) != required_params:
                    raise ValueError(f"Parameters for model {model_name} are incomplete.")
                model_key = _resolve_model_name(model_name)
                model_instance = MODEL_LIST[model_key](dim=self.dim, fov=self.fov, **model_params)
                mvis = torch.as_tensor(model_instance.sample_vis(uv, ttype=ttype), device=uv_t.device)
                mvis = mvis.reshape(-1).to(avis.dtype)
                if mvis.numel() != nvis:
                    raise ValueError(
                        f"Model {model_name} returned {mvis.numel()} visibilities, expected {nvis}."
                    )
                avis = avis + mvis
            return avis

    def key_params(self): # only implemented for xsring and xsringauss
        """Returns key parameters for the model
        
        Returns:
            Dictionary of key parameters for the model
        """
        for ind, model_name in enumerate(self.model_list):
            # returns first model's key params if multiple models are present
            if model_name in ['xsring', 'xsringauss']:
                model_params = {key.split('_', 1)[1]: value for key, value in self.params.items() if key.startswith(f"{ind}_")}
                required_params = set(_params_for_model(model_name))
                if set(model_params.keys()) != required_params:
                    raise ValueError(f"Parameters for model {model_name} do not match required parameters.")
                model_key = _resolve_model_name(model_name)
                model_instance = MODEL_LIST[model_key](dim=self.dim, fov=self.fov, **model_params)
                key_params = model_instance.key_params()
                return key_params
        return None