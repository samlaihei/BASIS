from BASIS.models import base

import bilby
import numpy as np
import torch
import itertools as it
import pandas as pd

class ModelLikelihood(bilby.Likelihood):
    def __init__(self, obs, model_names=None, img=None, imgdim=128, fov=225, count='min',  dterms={'vis':100},
                 static_noise_floor=0, noise_frac=0, noise_factor=1, avg_time=0, scan_avg=False,
                 obs_addAll=True):
        """Initialize the ModelLikelihood class.

        Parameters
        ----------
        obs : eh.obsdata.ObsData
            The observational data.
        model_names : list of str, optional
            The names of the models to use (default is None).
        img : eht.image.Image, optional
            The image data to use for the likelihood (default is None).
        imgdim : int, optional
            The image dimension (default is 128).
        fov : float, optional
            The field of view in microarcseconds (default is 225).
        count : str, optional
            'min' to use minimum count of triangles, 'max' to use maximum (default is 'min').
        static_noise_floor : float, optional
            The static noise floor to be added in quadrature (default is 0).
        noise_frac : float, optional
            The fraction of additional noise to be added (default is 0).
        noise_factor : float, optional
            The factor to scale the noise (default is 1).
        obs_addAll : bool, optional
            Whether to add all data terms to the obs data table (default is True).
        dterms : dict, optional
            The data terms to include in the likelihood and their weighting (default is visibility data).
        """

        super().__init__()
        self.model_names = model_names
        self.obs = obs
        self.imgdim = imgdim
        self.fov = fov
        self.count = count
        self.static_noise_floor = static_noise_floor
        self.noise_frac = noise_frac
        self.noise_factor = noise_factor
        self.img = img
        self.avg_time = avg_time
        self.scan_avg = scan_avg

        if obs_addAll:
            try:
                obs.add_all(count=self.count)
                print("All data terms added successfully.")
            except Exception as e:
                print(f"Error adding all data terms: {e}. Adding required data terms.")
                obs.add_bispec(count=self.count)
                obs.add_cphase(count=self.count)
                obs.add_camp(count=self.count)
                print("Required data terms added successfully.")

        self.cached_data = {}
        self.cached_noise = {}
        self.uv = {}

        self.dterms = dterms

        # Closure invariants
        self.normpower = 2
        
        if self.model_names is not None:
            self.model = base.BaseModel(model_list=self.model_names, dim=self.imgdim, fov=self.fov)
            self.params = self.model.params

    @staticmethod
    def _is_torch(x):
        return torch.is_tensor(x)

    @staticmethod
    def _to_torch(x, like=None, complex_ok=False):
        if torch.is_tensor(x):
            return x
        if like is not None and torch.is_tensor(like):
            if complex_ok:
                dtype = like.dtype
            else:
                dtype = like.real.dtype if like.is_complex() else like.dtype
            return torch.as_tensor(x, dtype=dtype, device=like.device)
        return torch.as_tensor(x)

    def clear_cached_data(self):
        self.cached_data = {}
        self.cached_noise = {}
        self.uv = {}
        if hasattr(self, "_coh_avg_plan_cache"):
            del self._coh_avg_plan_cache
        return

    def replace_obs_vis(self, new_vis, new_sigma, obs_addAll=False):
        """
        Replace only the observed visibilities and their associated uncertainties in the observation data.

        Parameters
        ----------
        new_vis : array-like
            The new observed visibilities to replace in the observation data.
        new_sigma : array-like
            The new uncertainties associated with the observed visibilities.
        obs_addAll : bool, optional
            Whether to add all data terms to the observation after replacing the visibilities. Default is False.
        """

        # Check shape matches obs.data['vis']
        if new_vis.shape != self.obs.data['vis'].shape:
            raise ValueError("Shape of new_vis does not match the shape of the observed visibilities.")
        if new_sigma.shape != self.obs.data['sigma'].shape:
            raise ValueError("Shape of new_sigma does not match the shape of the observed uncertainties.")

        self.obs.data['vis'] = new_vis
        self.obs.data['sigma'] = new_sigma

        if obs_addAll:
            self.obs.add_all(count=self.count)

        self.clear_cached_data()

        return 

    def _sample_vis_batched(self, model, uv_group, ttype='analytical'):
        """Evaluate multiple baseline UV tracks in a single model.sample_vis call."""
        uv_group = np.asarray(uv_group)
        if uv_group.ndim != 3 or uv_group.shape[1] != 2:
            raise ValueError("Expected uv_group shape (n_baselines, 2, n_points).")

        n_baselines = uv_group.shape[0]
        n_points = uv_group.shape[2]
        uv_stacked = uv_group.transpose(1, 0, 2).reshape(2, n_baselines * n_points)
        vis_stacked = model.sample_vis(uv_stacked, ttype=ttype)

        if self._is_torch(vis_stacked):
            return [vis_stacked[i * n_points:(i + 1) * n_points] for i in range(n_baselines)]

        vis_arr = np.asarray(vis_stacked)
        return [vis_arr[i * n_points:(i + 1) * n_points] for i in range(n_baselines)]

    def _gaussian_loglike(self, data, model_data, sigma):
        if self._is_torch(data) or self._is_torch(model_data) or self._is_torch(sigma):
            like = model_data if self._is_torch(model_data) else data
            data_t = self._to_torch(data, like=like, complex_ok=True)
            model_t = self._to_torch(model_data, like=data_t, complex_ok=True)
            residual_t = data_t - model_t
            sigma_t = self._to_torch(sigma, like=residual_t, complex_ok=False)
            return -0.5 * torch.sum(torch.abs(residual_t / sigma_t) ** 2 + torch.log(2 * torch.pi * sigma_t ** 2))
        residual = data - model_data
        return -0.5 * np.sum(np.abs((residual / sigma) ** 2) + np.log(2 * np.pi * sigma ** 2))

    def _gaussian_redchi2(self, data, model_data, sigma):
        if self._is_torch(data) or self._is_torch(model_data) or self._is_torch(sigma):
            like = model_data if self._is_torch(model_data) else data
            data_t = self._to_torch(data, like=like, complex_ok=True)
            model_t = self._to_torch(model_data, like=data_t, complex_ok=True)
            residual_t = data_t - model_t
            sigma_t = self._to_torch(sigma, like=residual_t, complex_ok=False)
            return torch.sum(torch.abs(residual_t / sigma_t) ** 2)/len(data_t)
        residual = data - model_data
        return np.sum(np.abs(residual / sigma) ** 2)/len(data)

    def _covariance_redchi2(self, data, model_data, cov):
        if self._is_torch(data) or self._is_torch(model_data) or self._is_torch(cov):
            like = model_data if self._is_torch(model_data) else data
            data_t = self._to_torch(data, like=like, complex_ok=True)
            model_t = self._to_torch(model_data, like=data_t, complex_ok=True)
            residual_t = data_t - model_t
            cov_t = self._to_torch(cov, like=residual_t, complex_ok=False)
            x_t = torch.linalg.solve(cov_t, residual_t)
            chi2_t = torch.sum(residual_t.conj() * x_t).real
            return chi2_t / len(data)
        residual = data - model_data
        x = np.linalg.solve(cov, residual)
        chi2 = np.vdot(residual, x).real 
        return chi2 / len(data)

    def advariants_to_ci_torch(self, advariants, times):
        """Normalise advariants to closure invariants with torch operations."""
        advariants_t = self._to_torch(advariants, complex_ok=True)
        times_t = self._to_torch(times, like=advariants_t.real)

        time_axis = torch.cat([times_t, times_t], dim=0)
        values = torch.cat([advariants_t.real, advariants_t.imag], dim=0)
        out = torch.empty_like(values)

        for t in torch.unique(times_t):
            mask = (time_axis == t)
            vals_t = values[mask]
            norm = torch.sum(torch.abs(vals_t) ** self.normpower).pow(1.0 / self.normpower)
            out[mask] = vals_t / norm

        return out


    def get_model(self, parameters):
        """Get the model instance based on the model name and parameters.

        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        model : object
            The model instance.
        """
        self.model.params = parameters
        return self.model

    def reduced_chi2(self, parameters, dtype='vis'):
        """Calculate the reduced chi-squared for the given parameters.
        
        Parameters
        ----------
        parameters : dict
            The model parameters.
        dtype : str, optional
            The data type to use for the reduced chi-squared calculation ('image', 'vis', 'visamp', 'bispec', 'cphase', 'camp', 'logcamp', 'ci'). Default is 'vis'.

        Returns
        -------
        redchi2 : float
            The reduced chi-squared.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype=dtype)
        return self._gaussian_redchi2(data, model_data, sigma)

    def log_likelihood(self, parameters):
        """Calculate the total log-likelihood for the given parameters.
        
        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        lnlike : float
            The total log-likelihood.
        """
        lnlike = 0
        for dterm, weight in self.dterms.items():
            add_lnlike = 0
            if dterm == 'image' and self.img is not None:
                add_lnlike = self.log_likelihood_image(parameters)
            elif dterm == 'vis':
                add_lnlike = self.log_likelihood_vis(parameters)
            elif dterm == 'visamp':
                add_lnlike = self.log_likelihood_visamp(parameters)
            elif dterm == 'bispec':
                add_lnlike = self.log_likelihood_bispec(parameters)
            elif dterm == 'cphase':
                add_lnlike = self.log_likelihood_cphase(parameters)
            elif dterm == 'camp':
                add_lnlike = self.log_likelihood_camp(parameters)
            elif dterm == 'logcamp':
                add_lnlike = self.log_likelihood_logcamp(parameters)
            elif dterm == 'ci':
                add_lnlike = self.log_likelihood_ci(parameters)

            add_lnlike *= weight/100
            lnlike += add_lnlike
        return lnlike

    def sample_data(self, parameters=None, dtype='vis', ttype='analytical', min_sep_time=0,
                    avg_time=None, scan_avg=None):
        """Sample the model data for the given parameters and data type.
        
        Parameters
        ----------
        parameters : dict
            The model parameters.
        dtype : str, optional
            The data type to sample ('image', 'vis', 'visamp', 'bispec', 'cphase', 'camp', 'logcamp', 'ci'). Default is 'vis'.
        ttype : str, optional
            The type of sampling ('analytical' or 'dft'). Default is 'analytical'.
        min_sep_time : float, optional
            The minimum separation time in seconds between triangles to be considered (default is 0). Only works for ci data type.
        avg_time : float, optional
            The averaging time in seconds for the data. Default is 0 (no averaging).
        scan_avg : bool, optional
            Whether to perform scan averaging. Default is False.
        avg_time : float, optional
            The averaging time in seconds for the data. Default is self.avg_time.
        scan_avg : bool, optional
            Whether to perform scan averaging. Default is self.scan_avg.

        Returns
        -------
        data : np.ndarray
            The observed data.
        sigma : np.ndarray
            The noise standard deviation.
        model_data : np.ndarray
            The model data.
        """
        if parameters is not None:
            model = self.get_model(parameters)
        if avg_time is None:
            avg_time = self.avg_time
        if scan_avg is None:
            scan_avg = self.scan_avg
        if avg_time != self.avg_time or scan_avg != self.scan_avg:
            self.scan_avg = scan_avg
            self.avg_time = avg_time
            self.clear_cached_data()
        
        data = None
    
        if dtype == 'image' and self.img is not None:
            if dtype not in self.cached_data:
                self.cached_data[dtype] = self.img
                self.cached_noise['sigma'+dtype] = np.ones_like(self.img._imdict['I'])*1e-4  # Assuming uniform noise for image
            if parameters is not None:
                data = model.sky_map()
        
        elif dtype == 'vis':
            if dtype not in self.cached_data:
                temp_obsdata = self.obs.data.copy()
                base_cols = [name for name in temp_obsdata.dtype.names if not any(k in name for k in ['vis', 'sigma'])]
                interleaved_cols = []
                for q in ['', 'q', 'u', 'v']:
                    interleaved_cols.extend([f'{q}vis', f'{q}sigma'])
                new_order = base_cols + interleaved_cols
                new_dtype = np.dtype([(name, temp_obsdata.dtype[name]) for name in new_order])
                reordered_data = np.empty(temp_obsdata.shape, dtype=new_dtype)
                for name in new_order:
                    reordered_data[name] = temp_obsdata[name]

                temp_obsdata = reordered_data
                temp_obsdata = self.coh_avg_ndarray(temp_obsdata, avg_time=avg_time, scan_avg=scan_avg)
                self.cached_data[dtype] = temp_obsdata['vis']
                self.cached_noise['sigma'+dtype] = temp_obsdata['sigma']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([temp_obsdata['u'], temp_obsdata['v']])
            if parameters is not None:
                data = model.sample_vis(self.uv[dtype], ttype=ttype)
            
        elif dtype == 'visamp':
            if dtype not in self.cached_data:                
                obsdata = self.coh_avg_ndarray(self.obs.amp, avg_time=avg_time, scan_avg=scan_avg)
                self.cached_data[dtype] = np.abs(obsdata['amp'])
                self.cached_noise['sigma'+dtype] = obsdata['sigma']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([obsdata['u'], obsdata['v']])
            if parameters is not None:
                vis_data = model.sample_vis(self.uv[dtype], ttype=ttype)
                data = torch.abs(vis_data) if self._is_torch(vis_data) else np.abs(vis_data)
        
        elif dtype == 'bispec':
            if dtype not in self.cached_data:
                obsdata = self.coh_avg_ndarray(self.obs.bispec, avg_time=avg_time, scan_avg=scan_avg)
                self.cached_data[dtype] = obsdata['bispec']
                self.cached_noise['sigma'+dtype] = obsdata['sigmab']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[obsdata['u1'], obsdata['v1']],
                                [obsdata['u2'], obsdata['v2']],
                                [obsdata['u3'], obsdata['v3']]])
            if parameters is not None:
                vis1, vis2, vis3 = self._sample_vis_batched(model, self.uv[dtype], ttype=ttype)
                data = vis1 * vis2 * vis3
        
        elif dtype == 'cphase':
            if dtype not in self.cached_data:
                obsdata = self.coh_avg_ndarray(self.obs.cphase, avg_time=avg_time, scan_avg=scan_avg)
                self.cached_data[dtype] = obsdata['cphase']*np.pi/180
                self.cached_noise['sigma'+dtype] = obsdata['sigmacp']*np.pi/180
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[obsdata['u1'], obsdata['v1']],
                                [obsdata['u2'], obsdata['v2']],
                                [obsdata['u3'], obsdata['v3']]])
            if parameters is not None:
                vis1, vis2, vis3 = self._sample_vis_batched(model, self.uv[dtype], ttype=ttype)
                bispec = vis1 * vis2 * vis3
                data = torch.angle(bispec) if self._is_torch(bispec) else np.angle(bispec)
        
        elif dtype == 'camp':
            if dtype not in self.cached_data:
                obsdata = self.coh_avg_ndarray(self.obs.camp, avg_time=avg_time, scan_avg=scan_avg)
                self.cached_data[dtype] = obsdata['camp']
                self.cached_noise['sigma'+dtype] = obsdata['sigmaca']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[obsdata['u1'], obsdata['v1']],
                                [obsdata['u2'], obsdata['v2']],
                                [obsdata['u3'], obsdata['v3']],
                                [obsdata['u4'], obsdata['v4']]])
            if parameters is not None:
                vis1, vis2, vis3, vis4 = self._sample_vis_batched(model, self.uv[dtype], ttype=ttype)
                data = vis1 * vis2 / (vis3 * vis4)
        
        elif dtype == 'logcamp':
            if dtype not in self.cached_data:
                obsdata = self.coh_avg_ndarray(self.obs.logcamp, avg_time=avg_time, scan_avg=scan_avg)
                self.cached_data[dtype] = obsdata['camp']
                self.cached_noise['sigma'+dtype] = obsdata['sigmaca']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[obsdata['u1'], obsdata['v1']],
                                [obsdata['u2'], obsdata['v2']],
                                [obsdata['u3'], obsdata['v3']],
                                [obsdata['u4'], obsdata['v4']]])
            if parameters is not None:
                vis1, vis2, vis3, vis4 = self._sample_vis_batched(model, self.uv[dtype], ttype=ttype)
                if any(self._is_torch(v) for v in [vis1, vis2, vis3, vis4]):
                    data = torch.log(vis1) + torch.log(vis2) - torch.log(vis3) - torch.log(vis4)
                else:
                    data = np.log(vis1) + np.log(vis2) - np.log(vis3) - np.log(vis4)
        
        elif dtype == 'ci':
            if dtype not in self.cached_data:
                adv, adv_noise, adv_metadata = self.calc_advariants(count=self.count, min_sep_time=min_sep_time)
                real_ci, real_ci_sigma = self.advariants_to_ci(adv, adv_metadata['time'], adv_noise)
                # real_ci_sigma = self.estimate_noise_ci() # MC-style estimation of CI noise, assuming independent Gaussian noise on visibilities

                adv_metadata['ci'], adv_metadata['sigma_ci'] = self.ci_real_to_complex(real_ci, real_ci_sigma)
                self.ci = adv_metadata
                self.ci = self.coh_avg_ndarray(self.ci, avg_time=avg_time, scan_avg=scan_avg)
                self.cached_data[dtype], self.cached_noise['sigma'+dtype] = self.ci_real_to_complex(self.ci['ci'], self.ci['sigma_ci'], invert=True)
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[self.ci['u1'], self.ci['v1']],
                                [self.ci['u2'], self.ci['v2']],
                                [self.ci['u3'], self.ci['v3']]])

            if parameters is not None:
                vis1, vis2, vis3 = self._sample_vis_batched(model, self.uv[dtype], ttype=ttype)
                if any(self._is_torch(v) for v in [vis1, vis2, vis3]):
                    advariants = vis1 * torch.conj(vis2).pow(-1) * vis3
                    data = self.advariants_to_ci_torch(advariants, self.ci['time'])
                else:
                    advariants = vis1 * np.conj(vis2)**(-1) * vis3
                    data = self.advariants_to_ci(advariants, self.ci['time'])[0]

        return self.cached_data[dtype], self.cached_noise['sigma'+dtype], data

    def log_likelihood_image(self, parameters):
        """Calculate the log-likelihood for image data.

        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        logL : float
            The log-likelihood for image data.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype='image')
        model = data.copy()
        model._imdict['I'] = model_data.flatten()
        model = data.align_images([model])[0][0]
        data_arr = data._imdict['I'].flatten()/data._imdict['I'].sum()
        model_data = model._imdict['I'].flatten()/model._imdict['I'].sum() 

        # data_arr = data._imdict['I'].flatten()/data._imdict['I'].sum()
        # model_data = model_data.flatten()/model_data.sum()

        residual = data_arr - model_data
        logL = -0.5 * np.sum(np.abs((residual / sigma) ** 2) + np.log(2 * np.pi * sigma ** 2))
        return logL

    def log_likelihood_vis(self, parameters):
        """Calculate the log-likelihood for visibility data.

        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        logL : float
            The log-likelihood for visibility data.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype='vis', ttype='analytical')
        return self._gaussian_loglike(data, model_data, sigma)

    def log_likelihood_visamp(self, parameters):
        """Calculate the log-likelihood for visibility amplitude data.

        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        logL : float
            The log-likelihood for visibility amplitude data.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype='visamp', ttype='analytical')
        return self._gaussian_loglike(data, model_data, sigma)

    def log_likelihood_bispec(self, parameters):
        """Calculate the log-likelihood for bispectrum data.
        
        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        logL : float
            The log-likelihood for bispectrum data.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype='bispec', ttype='analytical')
        return self._gaussian_loglike(data, model_data, sigma)

    def log_likelihood_cphase(self, parameters):
        """Calculate the log-likelihood for closure phase data.

        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        logL : float
            The log-likelihood for closure phase data.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype='cphase', ttype='analytical')
        if self._is_torch(model_data):
            data_t = self._to_torch(data, like=model_data)
            sigma_t = self._to_torch(sigma, like=model_data)
            residual = 1 - torch.cos(data_t - model_data)
            return -0.5 * torch.sum(residual / sigma_t ** 2 + torch.log(2 * torch.pi * sigma_t ** 2))
        residual = 1 - np.cos(data - model_data)
        return -0.5 * np.sum(residual / sigma ** 2 + np.log(2 * np.pi * sigma ** 2))

    def log_likelihood_camp(self, parameters):
        """Calculate the log-likelihood for closure amplitude data.

        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        logL : float
            The log-likelihood for closure amplitude data.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype='camp', ttype='analytical')
        return self._gaussian_loglike(data, model_data, sigma)

    def log_likelihood_logcamp(self, parameters):
        """Calculate the log-likelihood for log closure amplitude data.

        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        logL : float
            The log-likelihood for log closure amplitude data.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype='logcamp', ttype='analytical')
        return self._gaussian_loglike(data, model_data, sigma)

    def log_likelihood_ci(self, parameters): 
        """Calculate the log-likelihood for closure invariant data.

        Parameters
        ----------
        parameters : dict
            The model parameters.

        Returns
        -------
        logL : float
            The log-likelihood for closure invariant data.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype='ci', ttype='analytical')
        return self._gaussian_loglike(data, model_data, sigma)

    def augment_sigma(self, data, sigma):
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
        augmented_sigma = np.sqrt(sigma**2 + (self.noise_frac * np.abs(data))**2 + self.static_noise_floor**2) * self.noise_factor
        return augmented_sigma

    def tri_minimal_set(self, sites, tarr, tkey):
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

    def calc_advariants(self, N=1, add_noise=False, count='min', min_sep_time=0):
        """Calculate the advariants from the observational data.

        Parameters
        ----------
        N : int, optional
            The number of advariant samples to calculate (default is 1).
        add_noise : bool, optional
            Whether to add noise to the visibilities (default is False).
        count : str, optional
            'min' to use minimum count of triangles, 'max' to use maximum (default is 'min').
        min_sep_time : float, optional
            The minimum separation time in seconds between triangles to be considered (default is 0).

        Returns
        -------
        advariants : np.ndarray
            The calculated advariants.
        advariants_noise : np.ndarray
            The propagated advariant uncertainties.
        adv_metadata : np.ndarray
            Structured metadata for each advariant sample.
        """
        tlist = self.obs.tlist(conj=True)
        if not add_noise:
            N = 1

        adv_metadata_dtype = np.dtype([
            ('time', '<f8'),
            ('t1', '<U32'),
            ('t2', '<U32'),
            ('t3', '<U32'),
            ('u1', '<f8'),
            ('v1', '<f8'),
            ('u2', '<f8'),
            ('v2', '<f8'),
            ('u3', '<f8'),
            ('v3', '<f8'),
            ('adv', '<c16'),
            ('sigma', '<f8'),
            ('ci', '<c16'),
            ('sigma_ci', '<f8'),
        ])
        adv_metadata_rows = []
        triad_time_dict = {}

        v1s, v2s, v3s = [], [], []
        s1s, s2s, s3s = [], [], []
        metadata_entries = []

        for tdata in tlist:
            tdata = tdata.astype(self.obs.data.dtype)
            sites = list(set(np.hstack((tdata['t1'], tdata['t2']))))

            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat

            if count == 'max':
                tris = np.sort(list(it.combinations(sites, 3)))
            else:
                tris = self.tri_minimal_set(sites, self.obs.tarr, self.obs.tkey)

            for tri in tris:
                try:
                    # Select triangle entries in the data dictionary
                    l1 = l_dict[(tri[0], tri[1])]
                    l2 = l_dict[(tri[1], tri[2])]
                    l3 = l_dict[(tri[2], tri[0])]
                except KeyError:
                    # If any baseline is missing, skip this triangle
                    continue

                triad_time = tdata['time'][0]
                if tri not in triad_time_dict:
                    triad_time_dict[tri] = [triad_time]
                else:
                    if abs(triad_time - triad_time_dict[tri][-1]) < min_sep_time/3600:
                        continue
                    triad_time_dict[tri].append(triad_time)

                v1s.append(l1['vis'])
                v2s.append(l2['vis'])
                v3s.append(l3['vis'])
                s1s.append(l1['sigma'])
                s2s.append(l2['sigma'])
                s3s.append(l3['sigma'])
                metadata_entries.append((
                    triad_time,
                    tri[0],
                    tri[1],
                    tri[2],
                    l1['u'],
                    l1['v'],
                    l2['u'],
                    l2['v'],
                    l3['u'],
                    l3['v'],
                ))

        if len(v1s) == 0:
            advariants = np.empty((N, 0), dtype=np.complex128)
            advariants_noise = np.empty((N, 0), dtype=np.float64)
            adv_metadata = np.array(adv_metadata_rows, dtype=adv_metadata_dtype)
            return np.squeeze(advariants), np.squeeze(advariants_noise)/np.sqrt(2), adv_metadata

        v1, v2, v3 = [np.asarray(x, dtype=np.complex128) for x in (v1s, v2s, v3s)]
        s1, s2, s3 = [np.asarray(x, dtype=np.float64) for x in (s1s, s2s, s3s)]

        if add_noise:
            noise_v1, noise_v2, noise_v3 = [np.random.normal(0, s, size=(N, len(v))) + 1j * np.random.normal(0, s, size=(N, len(v))) for s, v in zip((s1, s2, s3), (v1, v2, v3))]
            v1, v2, v3 = v1[None, :] + noise_v1, v2[None, :] + noise_v2, v3[None, :] + noise_v3

        adv = v1 * np.conj(v2)**(-1) * v3
        adv_noise = np.sqrt((s1[None, :] / np.abs(v1))**2 + (s2[None, :] / np.abs(v2))**2 + (s3[None, :] / np.abs(v3))**2) * np.abs(adv)

        advariants = np.asarray(adv, dtype=np.complex128)
        advariants_noise = np.asarray(adv_noise, dtype=np.float64)
        adv0 = advariants[0] if advariants.ndim > 1 else advariants
        adv_noise0 = advariants_noise[0] if advariants_noise.ndim > 1 else advariants_noise
        advariants = np.squeeze(np.asarray(advariants).reshape(N, -1))
        advariants_noise = np.squeeze(np.asarray(advariants_noise).reshape(N, -1)) / np.sqrt(2)

        adv_metadata_rows = [
            (*entry, adv0[idx], adv_noise0[idx], np.nan, np.nan)
            for idx, entry in enumerate(metadata_entries)
        ]
        adv_metadata = np.array(adv_metadata_rows, dtype=adv_metadata_dtype)

        return advariants, advariants_noise, adv_metadata
    
    def advariants_to_ci(self, advariants, times, advariants_noise=None):
        """Normalise advariants to closure invariants using uncertainty propagation.
        
        Uses the uncertainties package for automatic error propagation if available,
        otherwise falls back to analytical error propagation. Assumes independent Gaussian noise on the advariants.

        Parameters
        ----------
        advariants : np.ndarray
            The advariants to normalise. May be a single 1D array of length M or a
            stacked 2D array of shape (N, M) for multiple noise realisations.
        times : np.ndarray
            The times corresponding to the advariants. Must be 1D with length M.
        advariants_noise : np.ndarray, optional
            The noise associated with the advariants. Must match the shape of advariants.

        Returns
        -------
        real_valued_advariants : np.ndarray
            The normalised closure invariants, preserving the input sample shape.
        real_valued_advariants_noise : np.ndarray
            The normalised closure invariants noise, if requested.
        """
        advariants = np.asarray(advariants)
        squeeze_output = False
        if advariants.ndim == 1:
            advariants = advariants[np.newaxis, :]
            squeeze_output = True
        elif advariants.ndim != 2:
            raise ValueError("advariants must be 1D or 2D")

        times = np.asarray(times)
        if times.ndim != 1 or times.shape[0] != advariants.shape[1]:
            raise ValueError("times must be 1D with the same length as the advariant samples")

        if advariants_noise is not None:
            advariants_noise = np.asarray(advariants_noise)
            if advariants_noise.ndim == 1:
                advariants_noise = advariants_noise[np.newaxis, :]
            if advariants_noise.shape != advariants.shape:
                raise ValueError("advariants_noise must match the shape of advariants")

        time_axis = np.concatenate([times, times], axis=0)
        real_valued_advariants = np.concatenate([advariants.real, advariants.imag], axis=1)

        if advariants_noise is not None:
            real_valued_advariants_noise = np.concatenate([advariants_noise, advariants_noise], axis=1)

        unique_times = np.unique(times)
        norm_per_time = np.empty((advariants.shape[0], len(unique_times)), dtype=float)
        norm_noise_per_time = np.empty((advariants.shape[0], len(unique_times)), dtype=float) if advariants_noise is not None else None

        for j, t in enumerate(unique_times):
            mask = (time_axis == t)
            values = real_valued_advariants[:, mask]
            sum_norm_p = np.sum(np.abs(values)**self.normpower, axis=1)
            norm_per_time[:, j] = sum_norm_p**(1/self.normpower)

            if advariants_noise is not None:
                sigmas = real_valued_advariants_noise[:, mask]
                dnorm_dx = (np.abs(values)**(self.normpower - 1) * np.sign(values) /
                            sum_norm_p[:, None]**(1 - 1/self.normpower))
                norm_noise_per_time[:, j] = np.sqrt(np.sum((dnorm_dx * sigmas)**2, axis=1))

        norm_map = np.empty_like(real_valued_advariants)
        for j, t in enumerate(unique_times):
            mask = (time_axis == t)
            norm_map[:, mask] = norm_per_time[:, j:j+1]

        real_valued_advariants_norm = real_valued_advariants / norm_map

        if advariants_noise is not None:
            real_valued_advariants_noise_norm = np.empty_like(real_valued_advariants_noise)
            for j, t in enumerate(unique_times):
                mask = (time_axis == t)
                value_unnormalized = real_valued_advariants[:, mask]
                sigma_x = real_valued_advariants_noise[:, mask]
                sigma_norm = norm_noise_per_time[:, j:j+1]
                norm = norm_per_time[:, j:j+1]
                variance = (sigma_x / norm)**2 + (value_unnormalized * sigma_norm / norm**2)**2
                real_valued_advariants_noise_norm[:, mask] = np.sqrt(variance)
            if squeeze_output:
                return real_valued_advariants_norm[0], real_valued_advariants_noise_norm[0]
            return real_valued_advariants_norm, real_valued_advariants_noise_norm

        if squeeze_output:
            return real_valued_advariants_norm[0], None
        return real_valued_advariants_norm, None
    
    
    def estimate_noise_ci(self, N=1000): # MC-style, assuming Gaussian noise on visibilities and propagating to closure invariants
        """Estimate the noise for closure invariant data.

        Returns
        -------
        sigma_ci : np.ndarray
            The estimated noise for closure invariant data.
        """

        advariants_noisy, _, adv_metadata = self.calc_advariants(N=N, add_noise=True, count=self.count)
        ci_noisy_samples, _ = self.advariants_to_ci(advariants_noisy, adv_metadata['time'])
        sigma_ci = np.std(ci_noisy_samples, axis=0)
        return sigma_ci

    def plot_all(self, parameters=None, save_path=None, avg_time=None, scan_avg=None, fig=None, axs=None, color='b'):
        """Plot the data and model for all data types.

        Parameters
        ----------
        parameters : dict, optional
            The model parameters (default is None).
        save_path : str, optional
            The path to save the plot (default is None).
        avg_time : float, optional
            The averaging time for the data (default is None).
        scan_avg : bool, optional
            Whether to perform scan averaging (default is None).
        fig : matplotlib.figure.Figure, optional
            The figure to plot on (default is None, which creates a new figure).
        axs : np.ndarray, optional
            The axes to plot on (default is None, which creates new axes).
        color : str, optional
            The color for the plot (default is 'b').
        """
        import matplotlib.pyplot as plt

        if avg_time is None:
            avg_time = self.avg_time
        if scan_avg is None:
            scan_avg = self.scan_avg

        if fig is None or axs is None or axs.shape != (2, 4):
            fig, axs = plt.subplots(2, 4, figsize=(16,12))
        errorbar_alpha = 0.4

        dterms = {'Visibility Amplitude': 'visamp', 'Visibility Phase': 'vis', 'Bispectrum Amplitude': 'bispec',
                'Closure Phase': 'cphase', 'Closure Amplitude': 'camp', 'Log Closure Amplitude': 'logcamp', 'Closure Invariants Amplitude': 'ci', 'Closure Invariants Phase': 'ci',}
                # 'Closure Invariants Real': 'ci', 'Closure Invariants Imag': 'ci', 
                # 'log Closure Invariants Real': 'ci', 'log Closure Invariants Imag': 'ci'}
        ops = {'Visibility Amplitude': np.abs, 'Visibility Phase': np.angle, 'Bispectrum Amplitude': np.abs, 
            'Closure Phase': lambda x: x, 'Closure Amplitude': np.abs, 'Log Closure Amplitude': lambda x: x, 'Closure Invariants Amplitude': np.abs, 'Closure Invariants Phase': np.angle,}
            # 'Closure Invariants Real': np.real, 'Closure Invariants Imag': np.imag,
            # 'log Closure Invariants Real': lambda x: np.log(np.real(x)), 'log Closure Invariants Imag': lambda x: np.log(np.imag(x))}

        for i, (title, dtype) in enumerate(dterms.items()):
            data, sigma, model_data = self.sample_data(parameters, dtype=dtype, ttype='analytical', avg_time=avg_time, scan_avg=scan_avg)
            if dtype in ['visamp', 'vis']:
                uvdist = np.sqrt(self.uv[dtype][0]**2 + self.uv[dtype][1]**2)
            else:
                uvdist = [np.sqrt(self.uv[dtype][j][0]**2 + self.uv[dtype][j][1]**2) for j in range(len(self.uv[dtype]))]
                uvdist = np.sum(uvdist, axis=0)

            if title == 'Visibility Phase':
                sigma = np.real(sigma/np.abs(data))  # phase error approximation

            if dtype in ['ci']:
                data = data[:len(data)//2] + 1j*data[len(data)//2:]
                if parameters is not None:
                    model_data = model_data[:len(model_data)//2] + 1j*model_data[len(model_data)//2:]
                sigma = sigma[:len(sigma)//2]
                if title == 'Closure Invariants Phase':
                    sigma = np.real(sigma/np.abs(data))  # phase error approximation
                
            axs[i%2, i//2].errorbar(uvdist, ops[title](data), yerr=sigma, fmt='o', color=color, markersize=0, capsize=1, alpha=errorbar_alpha, label='Data')
            if parameters is not None:
                axs[i%2, i//2].scatter(uvdist, ops[title](model_data), color='red', s=5, label='Model')
            axs[i%2, i//2].set_title(title)
            if 'Phase' in title:
                axs[i%2, i//2].set_ylim(-np.pi*1.1, np.pi*1.1)
            if i == 0:
                axs[i%2, i//2].legend()

        fig.text(0.5, 0.04, 'Total uv-Distance (λ)', ha='center', va='center', fontsize=18)
        fig.text(0.08, 0.5, 'Observable', ha='center', va='center', rotation='vertical', fontsize=18)

        if save_path is not None:
            plt.savefig(save_path, dpi=200)

        return fig, axs
        
    def sample_all(self, parameters=None, ttype='analytical',
                   avg_time=None, scan_avg=None):
        """Sample the model data for all data types.

        Parameters
        ----------
        parameters : dict
            The model parameters.
        ttype : str, optional
            The type of sampling ('analytical' or 'dft'). Default is 'analytical'.
        avg_time : float, optional
            The averaging time in seconds for the data. Default is self.avg_time.
        scan_avg : bool, optional
            Whether to perform scan averaging. Default is self.scan_avg.

        Returns
        -------
        all_data : dict
            A dictionary containing the sampled data for each data type.
        all_sigma : dict
            A dictionary containing the sampled sigma for each data type.
        all_model_data : dict
            A dictionary containing the sampled model data for each data type.
        """
        all_data, all_sigma, all_model_data = {}, {}, {}
        dtypes = ['vis', 'visamp', 'bispec', 'cphase', 'camp', 'logcamp', 'ci']
        for dtype in dtypes:
            data, sigma, model_data = self.sample_data(parameters, dtype=dtype, ttype=ttype, avg_time=avg_time, scan_avg=scan_avg)
            all_data[dtype] = data
            all_sigma[dtype] = sigma
            all_model_data[dtype] = model_data
        return all_data, all_sigma, all_model_data

    def ci_real_to_complex(self, ci, ci_err, invert=False):
        """
        Convert a real-valued CI array and its error to a complex-valued CI array and its error.

        Parameters
        ----------
        ci : ndarray
            The real-valued CI array, where the first half represents the real part and the second half represents the imaginary part.
        ci_err : ndarray
            The error associated with the real-valued CI array.
        invert : bool, optional
            If True, convert from complex to real representation. Default is False.

        Returns
        -------
        complex_ci : ndarray
            The complex-valued CI array.
        complex_ci_err : ndarray
            The error associated with the complex-valued CI array.
        """
        if invert:
            ci = np.concatenate([ci.real, ci.imag])
            ci_err = np.concatenate([ci_err, ci_err])
            return ci, ci_err
        else:
            complex_ci = ci[:len(ci)//2] + 1j*ci[len(ci)//2:]
            ci_err = ci_err[:len(ci_err)//2]
            return complex_ci, ci_err

    def add_scans(self, time_array, dt=0.0165, margin=0.0001):
        """
        Given an array of time-stamps, identify and group scans based on time gaps.

        Parameters
        ----------
        time_array : array-like
            Array of time stamps.
        dt : float, optional
            Time gap threshold to define a new scan. Default is 0.0165.
        margin : float, optional
            Margin to add before and after each scan. Default is 0.0001.

        Returns
        -------
        scanlist : ndarray
            Array of start and end times for each scan, including margins.
        """
        
        times_uni = np.asarray(sorted(list(set(time_array))))
        scans = np.zeros_like(times_uni)
        scan_id = 0
        for cou in range(len(times_uni) - 1):
            scans[cou] = scan_id
            if (times_uni[cou + 1] - times_uni[cou] > dt):
                scan_id += 1
        scans[-1] = scan_id
        scanlist = np.asarray([np.asarray([
                                np.min(times_uni[scans == cou]) - margin,
                                np.max(times_uni[scans == cou]) + margin])
                                for cou in range(int(scans[-1]) + 1)])

        return scanlist

    def coh_avg_ndarray(self, data_array, avg_time=0, scan_avg=True, dt=0.0165, margin=0.0001):
        """
        Coherently average the data array over time or over scans.

        Parameters
        ----------
        data_array : ndarray
            The input data array to be averaged.
            ['time', 't1', 't2', ('t3', 't4'), ('data', 'sigma')]
        avg_time : float, optional
            The time interval over which to average the data. Default is 0 (no averaging).
        scan_avg : bool, optional
            Whether to perform scan averaging. Default is True.
        dt : float, optional
            Time gap threshold to define a new scan. Default is 0.0165.
        margin : float, optional
            Margin to add before and after each scan. Default is 0.0001.

        Returns
        -------
        avg_data : ndarray
            The coherently averaged data array.
        """

        if avg_time <= 0 and not scan_avg:
            return data_array

        dtype = data_array.dtype
        names = dtype.names

        tn_cols = [c for c in names if c.startswith('t') and c != 'time']
        sigma_cols = [c for c in names if 'sigma' in c]
        mean_cols = [c for c in names if c not in tn_cols + sigma_cols + ['time']]

        # ---------- cache init ----------
        if not hasattr(self, "_coh_avg_plan_cache"):
            self._coh_avg_plan_cache = {}
            self._coh_avg_plan_cache_base = None

        # Cache signature
        t = data_array['time']
        base_sig = (
            len(data_array),
            dtype.descr,
            tuple(tn_cols),
            t.dtype.str,
            float(t[0]) if len(t) else None,
            float(t[-1]) if len(t) else None,
        )

        if self._coh_avg_plan_cache_base != base_sig:
            self._coh_avg_plan_cache.clear()
            self._coh_avg_plan_cache_base = base_sig

        plan_key = (float(avg_time), bool(scan_avg), float(dt), float(margin))
        plan = self._coh_avg_plan_cache.get(plan_key)

        # ---------- build plan if signature not in cache ----------
        if plan is None:
            all_time = data_array['time']

            # Group rows by tn_cols
            if tn_cols:
                key_dtype = [(c, data_array.dtype[c]) for c in tn_cols]
                keys = np.empty(len(data_array), dtype=key_dtype)
                for c in tn_cols:
                    keys[c] = data_array[c]
                _, group_inv = np.unique(keys, return_inverse=True)
                n_groups = int(group_inv.max()) + 1
            else:
                group_inv = np.zeros(len(data_array), dtype=np.int64)
                n_groups = 1

            group_plans = []
            for gid in range(n_groups):
                grp_idx = np.flatnonzero(group_inv == gid)
                if grp_idx.size == 0:
                    continue

                # Stable sort by time
                ord_local = np.argsort(all_time[grp_idx], kind='mergesort')
                ord_idx = grp_idx[ord_local]
                tg = all_time[ord_idx]

                if scan_avg:
                    avg_interval = np.asarray(self.add_scans(tg, dt=dt, margin=margin))
                    starts = avg_interval[:, 0]
                    ends = avg_interval[:, 1]
                    b = np.searchsorted(starts, tg, side='right') - 1
                    valid = (b >= 0) & (tg <= ends[np.clip(b, 0, len(ends) - 1)])
                    if not np.any(valid):
                        continue
                    ord_idx = ord_idx[valid]
                    tg = tg[valid]
                    b = b[valid]
                    _, bin_inv = np.unique(b, return_inverse=True)
                else:
                    if avg_time == 0:
                        # one bin per unique timestamp
                        _, bin_inv = np.unique(tg, return_inverse=True)
                    else:
                        # dynamic-edge semantics (same as your old function)
                        avg_t = avg_time / 3600.0
                        bin_inv = np.zeros(tg.size, dtype=np.int64)
                        b = 0
                        bin_start = tg[0]
                        for i in range(1, tg.size):
                            if tg[i] - bin_start >= avg_t:
                                b += 1
                                bin_start = tg[i]
                            bin_inv[i] = b

                n_bins = int(bin_inv.max()) + 1
                counts = np.bincount(bin_inv, minlength=n_bins).astype(np.float64)

                group_plans.append({
                    "ord_idx": ord_idx,
                    "t_sorted": tg,           # used for fast time averaging
                    "bin_inv": bin_inv,
                    "counts": counts,
                    "n_bins": n_bins,
                    "tn_ref_idx": ord_idx[0], # representative row for tn cols
                })

            plan = {
                "group_plans": group_plans,
                "tn_cols": tn_cols,
                "mean_cols": mean_cols,
                "sigma_cols": sigma_cols,
                "dtype": dtype,
            }
            self._coh_avg_plan_cache[plan_key] = plan

        # ---------- apply plan to current values ----------
        out_chunks = []
        for gp in plan["group_plans"]:
            ord_idx = gp["ord_idx"]
            tg = gp["t_sorted"]
            bin_inv = gp["bin_inv"]
            counts = gp["counts"]
            n_bins = gp["n_bins"]

            chunk = np.empty(n_bins, dtype=dtype)

            # constant tn fields per group
            for c in tn_cols:
                chunk[c] = data_array[c][gp["tn_ref_idx"]]

            # average time per bin
            chunk['time'] = np.bincount(bin_inv, weights=tg, minlength=n_bins) / counts

            # mean reductions (complex-safe)
            for c in mean_cols:
                vals = data_array[c][ord_idx]
                if np.issubdtype(vals.dtype, np.complexfloating):
                    re = np.bincount(bin_inv, weights=vals.real, minlength=n_bins) / counts
                    im = np.bincount(bin_inv, weights=vals.imag, minlength=n_bins) / counts
                    chunk[c] = re + 1j * im
                else:
                    num = np.bincount(
                        bin_inv,
                        weights=vals.astype(np.float64, copy=False),
                        minlength=n_bins
                    )
                    chunk[c] = num / counts

            # sigma combine: sqrt(sum(sigma^2))/N
            for c in sigma_cols:
                sig = data_array[c][ord_idx]
                if np.issubdtype(sig.dtype, np.complexfloating):
                    sig2 = sig.real * sig.real + sig.imag * sig.imag
                else:
                    sigf = sig.astype(np.float64, copy=False)
                    sig2 = sigf * sigf
                rss = np.sqrt(np.bincount(bin_inv, weights=sig2, minlength=n_bins))
                chunk[c] = rss / counts

            out_chunks.append(chunk)

        # sort concatenated out_chunks by time
        out_chunks = np.concatenate(out_chunks)
        out_chunks = out_chunks[np.argsort(out_chunks['time'])]
        return out_chunks
