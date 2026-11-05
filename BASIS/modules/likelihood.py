from GCFit.models import base
from GCFit.modules import utils

import bilby
import numpy as np
import torch
import itertools as it

class ModelLikelihood(bilby.Likelihood):
    def __init__(self, obs, model_names=None, imgdim=128, fov=225, count='min',  dterms={'vis':100},
                 static_noise_floor=0, noise_frac=0, noise_factor=1, avg_time=0, scan_avg=False, uvdist_sections=None,
                 use_cov=False, cov_samples=1000):
        """Initialize the ModelLikelihood class.

        Parameters
        ----------
        obs : eh.obsdata.ObsData
            The observational data.
        model_names : list of str, optional
            The names of the models to use (default is None).
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
        dterms : dict, optional
            The data terms to include in the likelihood and their weighting (default is visibility data).
        use_cov : bool, optional
            Use covariance likelihoods for each data term (default is False).
            Visibilities use analytical diagonal covariance; other terms use MC.
            Covariances are separate per term, without cross-term correlations.
        cov_samples : int, optional
            Minimum MC draws (default 1000); at least five per real feature are
            used. Covariance is cached after averaging, filtering and augmentation.
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
        self.avg_time = avg_time
        self.scan_avg = scan_avg
        self.uvdist_sections = uvdist_sections

        self.cached_data = {}
        self.cached_noise = {}
        self.uv = {}

        self.dterms = dterms
        self.use_cov = use_cov
        if not isinstance(cov_samples, (int, np.integer)) or cov_samples < 2:
            raise ValueError("cov_samples must be an integer >= 2")
        self.cov_samples = cov_samples

        # Closure invariants
        self.normpower = 2
        
        if self.model_names is not None:
            self.model = base.BaseModel(model_list=self.model_names, dim=self.imgdim, fov=self.fov)
            self.params = self.model.params

    @staticmethod
    def _reorder_obsdata_ndarray(obsdata):
        temp_obsdata = obsdata.data.copy()
        base_cols = [name for name in temp_obsdata.dtype.names if not any(k in name for k in ['vis', 'sigma'])]
        interleaved_cols = []
        for q in ['', 'q', 'u', 'v']:
            interleaved_cols.extend(name for name in (f'{q}vis', f'{q}sigma')
                                    if name in temp_obsdata.dtype.names)
        new_order = base_cols + interleaved_cols
        new_dtype = np.dtype([(name, temp_obsdata.dtype[name]) for name in new_order])
        reordered_data = np.empty(temp_obsdata.shape, dtype=new_dtype)
        for name in new_order:
            reordered_data[name] = temp_obsdata[name]
        return reordered_data

    def clear_cached_data(self):
        self.cached_data = {}
        self.cached_noise = {}
        self.uv = {}
        for name in ('_coh_avg_plan_cache', '_ci_group_cache', '_uv_plan_cache',
                     '_gaussian_norm_cache', '_uv_mask_cache', '_camp_metadata'):
            if hasattr(self, name):
                delattr(self, name)
        return

    def replace_obs_vis(self, new_vis, new_sigma):
        """
        Replace only the observed visibilities and their associated uncertainties in the observation data.

        Parameters
        ----------
        new_vis : array-like
            The new observed visibilities to replace in the observation data.
        new_sigma : array-like
            The new uncertainties associated with the observed visibilities.
        """

        # Check shape matches obs.data['vis']
        if new_vis.shape != self.obs.data['vis'].shape:
            raise ValueError("Shape of new_vis does not match the shape of the observed visibilities.")
        if new_sigma.shape != self.obs.data['sigma'].shape:
            raise ValueError("Shape of new_sigma does not match the shape of the observed uncertainties.")

        self.obs.data['vis'] = new_vis
        self.obs.data['sigma'] = new_sigma

        self.clear_cached_data()

        return 

    def _sample_vis_batched(self, model, uv_group, ttype='analytical'):
        """Sample distinct UV coordinates once, then restore each baseline track."""
        uv_group = np.asarray(uv_group)
        if uv_group.ndim != 3 or uv_group.shape[1] != 2:
            raise ValueError("Expected uv_group shape (n_baselines, 2, n_points).")

        n_baselines = uv_group.shape[0]
        n_points = uv_group.shape[2]
        if not hasattr(self, '_uv_plan_cache'):
            self._uv_plan_cache = {}
        key = id(uv_group)
        plan = self._uv_plan_cache.get(key)
        if plan is None or not np.array_equal(plan[0], uv_group):
            uv_stacked = uv_group.transpose(1, 0, 2).reshape(2, -1)
            unique_uv, inverse = np.unique(uv_stacked.T, axis=0, return_inverse=True)
            plan = (uv_group.copy(), unique_uv.T.copy(), inverse)
            # One plan per shape bounds memory for callers supplying fresh arrays.
            self._uv_plan_cache = {k: v for k, v in self._uv_plan_cache.items()
                                   if v[0].shape != uv_group.shape}
            self._uv_plan_cache[key] = plan

        vis_unique = model.sample_vis(plan[1], ttype=ttype)
        if utils.is_torch(vis_unique):
            inverse = torch.as_tensor(plan[2], device=vis_unique.device)
            vis_stacked = vis_unique[inverse]
        else:
            vis_stacked = np.asarray(vis_unique)[plan[2]]
        return [vis_stacked[i * n_points:(i + 1) * n_points] for i in range(n_baselines)]

    def _gaussian_loglike(self, data, model_data, sigma):
        if utils.is_torch(data) or utils.is_torch(model_data) or utils.is_torch(sigma):
            like = model_data if utils.is_torch(model_data) else data
            data_t = utils.to_torch(data, like=like, complex_ok=True)
            model_t = utils.to_torch(model_data, like=data_t, complex_ok=True)
            residual_t = data_t - model_t
            sigma_t = utils.to_torch(sigma, like=residual_t, complex_ok=False)
            return -0.5 * torch.sum(torch.abs(residual_t / sigma_t) ** 2 + torch.log(2 * torch.pi * sigma_t ** 2))
        sigma = np.asarray(sigma)
        if not hasattr(self, '_gaussian_norm_cache'):
            self._gaussian_norm_cache = {}
        key = (sigma.shape, sigma.dtype.str)
        cached = self._gaussian_norm_cache.get(key)
        if cached is None or not np.array_equal(cached[0], sigma):
            cached = (sigma.copy(), np.log(2 * np.pi * sigma**2))
            self._gaussian_norm_cache[key] = cached
        residual = data - model_data
        return -0.5 * np.sum(np.abs((residual / sigma)**2) + cached[1])

    def _covariance_loglike(self, data, model_data, logcovdet, invcov):
        if utils.is_torch(data) or utils.is_torch(model_data) or utils.is_torch(logcovdet) or utils.is_torch(invcov):
            like = model_data if utils.is_torch(model_data) else data
            data_t = utils.to_torch(data, like=like, complex_ok=True)
            model_t = utils.to_torch(model_data, like=data_t, complex_ok=True)
            residual_t = data_t - model_t
            logcovdet_t = utils.to_torch(logcovdet, like=residual_t, complex_ok=False)
            invcov_t = utils.to_torch(invcov, like=residual_t, complex_ok=False)
            x_t = torch.matmul(invcov_t, residual_t)
            chi2_t = torch.sum(residual_t.conj() * x_t).real
            return -0.5 * (chi2_t + logcovdet_t + data_t.numel() * np.log(2 * np.pi))
        residual = data - model_data
        x = np.matmul(invcov, residual)
        chi2 = np.vdot(residual, x).real
        return -0.5 * (chi2 + logcovdet + np.size(data) * np.log(2 * np.pi))

    def _gaussian_redchi2(self, data, model_data, sigma):
        if utils.is_torch(data) or utils.is_torch(model_data) or utils.is_torch(sigma):
            like = model_data if utils.is_torch(model_data) else data
            data_t = utils.to_torch(data, like=like, complex_ok=True)
            model_t = utils.to_torch(model_data, like=data_t, complex_ok=True)
            residual_t = data_t - model_t
            sigma_t = utils.to_torch(sigma, like=residual_t, complex_ok=False)
            return torch.sum(torch.abs(residual_t / sigma_t) ** 2)/len(data_t)
        residual = data - model_data
        return np.sum(np.abs(residual / sigma) ** 2)/len(data)

    def _covariance_redchi2(self, data, model_data, invcov):
        if utils.is_torch(data) or utils.is_torch(model_data) or utils.is_torch(invcov):
            like = model_data if utils.is_torch(model_data) else data
            data_t = utils.to_torch(data, like=like, complex_ok=True)
            model_t = utils.to_torch(model_data, like=data_t, complex_ok=True)
            residual_t = data_t - model_t
            invcov_t = utils.to_torch(invcov, like=residual_t, complex_ok=False)
            x_t = torch.matmul(invcov_t, residual_t)
            chi2_t = torch.sum(residual_t.conj() * x_t).real
            return chi2_t / len(data)
        residual = data - model_data
        x = np.matmul(invcov, residual)
        chi2 = np.vdot(residual, x).real 
        return chi2 / len(data)

    def advariants_to_ci_torch(self, advariants, times):
        """Normalise advariants to closure invariants with torch operations."""
        advariants_t = utils.to_torch(advariants, complex_ok=True)
        times_t = utils.to_torch(times, like=advariants_t.real)

        values = torch.cat([advariants_t.real, advariants_t.imag], dim=0)
        _, groups = torch.unique(times_t, return_inverse=True)
        groups = torch.cat([groups, groups])
        sums = torch.zeros_like(times_t).scatter_add(0, groups, torch.abs(values)**self.normpower)
        norms = sums.pow(1.0 / self.normpower)
        return values / norms[groups]


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
            The data type to use for the reduced chi-squared calculation ('vis', 'visamp', 'bispec', 'cphase', 'camp', 'logcamp', 'ci'). Default is 'vis'.

        Returns
        -------
        redchi2 : float
            The reduced chi-squared.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype=dtype)
        if self.use_cov:
            if dtype == 'cphase':
                residual = (utils.to_torch(data, like=model_data) - model_data
                            if utils.is_torch(model_data) else data - model_data)
                if utils.is_torch(residual):
                    residual = torch.atan2(torch.sin(residual), torch.cos(residual))
                else:
                    residual = np.angle(np.exp(1j * residual))
                return self._covariance_redchi2(residual, residual * 0, self.cached_noise['invcov_'+dtype])
            return self._covariance_redchi2(utils.real_features(data), utils.real_features(model_data),
                                           self.cached_noise['invcov_'+dtype])
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
        dterm_func = {'vis': self.log_likelihood_vis,
                      'visamp': self.log_likelihood_visamp,
                      'bispec': self.log_likelihood_bispec,
                      'cphase': self.log_likelihood_cphase,
                      'camp': self.log_likelihood_camp,
                      'logcamp': self.log_likelihood_logcamp,
                      'ci': self.log_likelihood_ci}
        
        for dterm, weight in self.dterms.items():
            add_lnlike = 0
            loglike_func = dterm_func.get(dterm, None)
            if loglike_func is not None:
                add_lnlike = loglike_func(parameters)

            add_lnlike *= weight/100
            lnlike += add_lnlike
        return lnlike

    def sample_data(self, parameters=None, dtype='vis', ttype='analytical', min_sep_time=0,
                    avg_time=None, scan_avg=None, uvdist_sections=None):
        """Sample the model data for the given parameters and data type.
        
        Parameters
        ----------
        parameters : dict
            The model parameters.
        dtype : str, optional
            The data type to sample ('vis', 'visamp', 'bispec', 'cphase', 'camp', 'logcamp', 'ci'). Default is 'vis'.
        ttype : str, optional
            The type of sampling ('analytical' or 'dft'). Default is 'analytical'.
        min_sep_time : float, optional
            The minimum separation time in seconds between repeated triangles or quadrangles (default is 0).
        avg_time : float, optional
            The averaging time in seconds for the data. Default is 0 (no averaging).
        scan_avg : bool, optional
            Whether to perform scan averaging. Default is False.
        avg_time : float, optional
            The averaging time in seconds for the data. Default is self.avg_time.
        scan_avg : bool, optional
            Whether to perform scan averaging. Default is self.scan_avg.
        uvdist_sections : list of float, optional
            The sections of the uvdist to consider. Default is self.uvdist_sections.

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
        if uvdist_sections is None:
            uvdist_sections = self.uvdist_sections
        config = (self.count, float(min_sep_time), float(avg_time), bool(scan_avg),
                  None if uvdist_sections is None else tuple(map(tuple, uvdist_sections)),
                  bool(self.use_cov), self.cov_samples, self.noise_frac,
                  self.static_noise_floor, self.noise_factor)
        if getattr(self, '_sample_config', None) != config:
            self.clear_cached_data()
            self._sample_config = config

        data = None
        
        if dtype in ('vis', 'visamp'):
            if dtype not in self.cached_data:
                obsdata = self._reorder_obsdata_ndarray(self.obs)
                obsdata = self.coh_avg_ndarray(obsdata, avg_time=avg_time, scan_avg=scan_avg)
                values = obsdata['vis'] if dtype == 'vis' else np.abs(obsdata['vis'])
                self._cache_observed_data(dtype, values, obsdata['sigma'], obsdata,
                                          uvdist_sections)

            if parameters is not None:
                data = model.sample_vis(self.uv[dtype], ttype=ttype)
                if dtype == 'visamp':
                    data = torch.abs(data) if utils.is_torch(data) else np.abs(data)
                mask = self.filter_by_uvdist_ranges(self.uv[dtype], uvdist_sections)
                data = data[mask]

        elif dtype in ('bispec', 'cphase'):
            if dtype not in self.cached_data:
                metadata = self.calc_bispectrum(count=self.count, min_sep_time=min_sep_time)[2]
                self.bispec = self.coh_avg_ndarray(metadata, avg_time=avg_time, scan_avg=scan_avg)
                values, sigma = self.bispec['bispec'], self.bispec['sigma_bispec']
                if dtype == 'cphase':
                    amplitude = np.abs(values)
                    sigma = np.divide(sigma, amplitude, out=np.full(len(values), np.inf),
                                      where=amplitude > 0)
                    values = np.angle(values)
                self._cache_observed_data(dtype, values, sigma, self.bispec,
                                          uvdist_sections, n_baselines=3)

            if parameters is not None:
                vis1, vis2, vis3 = self._sample_vis_batched(model, self.uv[dtype], ttype=ttype)
                data = vis1 * vis2 * vis3
                if dtype == 'cphase':
                    data = torch.angle(data) if utils.is_torch(data) else np.angle(data)
                mask = self.filter_by_uvdist_ranges(self.uv[dtype], uvdist_sections)
                data = data[mask]

        elif dtype in ('camp', 'logcamp'):
            if dtype not in self.cached_data:
                if not hasattr(self, '_camp_metadata'):
                    self._camp_metadata = self.calc_camp(
                        count=self.count, min_sep_time=min_sep_time)[2]
                self.camp = self.coh_avg_ndarray(
                    self._camp_metadata, avg_time=avg_time, scan_avg=scan_avg)
                values, sigma = self.camp['camp'], self.camp['sigma_camp']
                if dtype == 'logcamp':
                    sigma = sigma / values
                    values = np.log(values)
                self._cache_observed_data(dtype, values, sigma, self.camp,
                                          uvdist_sections, n_baselines=4)

            if parameters is not None:
                vis = self._sample_vis_batched(model, self.uv[dtype], ttype=ttype)
                if any(utils.is_torch(v) for v in vis):
                    amplitudes = [torch.abs(v) for v in vis]
                    data = amplitudes[0] * amplitudes[1] / (amplitudes[2] * amplitudes[3])
                    if dtype == 'logcamp':
                        data = torch.log(data)
                else:
                    amplitudes = [np.abs(v) for v in vis]
                    data = amplitudes[0] * amplitudes[1] / (amplitudes[2] * amplitudes[3])
                    if dtype == 'logcamp':
                        data = np.log(data)
                mask = self.filter_by_uvdist_ranges(self.uv[dtype], uvdist_sections)
                data = data[mask]

        elif dtype == 'ci':
            if dtype not in self.cached_data:
                adv, adv_noise, metadata = self.calc_advariants(count=self.count, min_sep_time=min_sep_time)
                real_ci, real_ci_sigma = self.advariants_to_ci(adv, metadata['time'], adv_noise)
                metadata['ci'], metadata['sigma_ci'] = self.ci_real_to_complex(real_ci, real_ci_sigma)
                self.ci = self.coh_avg_ndarray(metadata, avg_time=avg_time, scan_avg=scan_avg)
                n = len(metadata)
                self._ci_raw_metadata = metadata
                self._ci_average = None
                if avg_time > 0 or scan_avg:
                    self._ci_average = self._ci_averaging_matrix(n, avg_time, scan_avg)
                    average = self._ci_average
                    average_squared = average**2
                    # Independent-error approximation for the analytical path.
                    real_ci_sigma = np.sqrt(np.concatenate([
                        average_squared @ real_ci_sigma[:n]**2,
                        average_squared @ real_ci_sigma[n:]**2]))
                values = np.concatenate([self.ci['ci'].real, self.ci['ci'].imag])
                self._cache_observed_data(dtype, values, real_ci_sigma, self.ci,
                                          uvdist_sections, n_baselines=3)

                # Model evaluations reuse the original UV coordinates.
                self._ci_raw_uv = np.array([[metadata['u1'], metadata['v1']],
                                           [metadata['u2'], metadata['v2']],
                                           [metadata['u3'], metadata['v3']]])

            if parameters is not None:
                metadata = self._ci_raw_metadata
                n = len(metadata)
                vis1, vis2, vis3 = self._sample_vis_batched(model, self._ci_raw_uv, ttype=ttype)
                if any(utils.is_torch(v) for v in [vis1, vis2, vis3]):
                    advariants = vis1 * torch.conj(vis2).pow(-1) * vis3
                    data = self.advariants_to_ci_torch(advariants, metadata['time'])
                else:
                    advariants = vis1 * np.conj(vis2)**(-1) * vis3
                    data = self.advariants_to_ci(advariants, metadata['time'])[0]
                if self._ci_average is not None:
                    average = self._ci_average
                    if utils.is_torch(data):
                        average = utils.to_torch(average, like=data)
                        data = torch.cat([average @ data[:n], average @ data[n:]])
                    else:
                        data = np.concatenate([average @ data[:n], average @ data[n:]])
                mask = self.filter_by_uvdist_ranges(self.uv[dtype], uvdist_sections)
                data = data[np.concatenate([mask, mask])]

        if self.use_cov and 'cov_'+dtype not in self.cached_noise:
            self._cache_observable_covariance(dtype, min_sep_time, uvdist_sections)
        return self.cached_data[dtype], self.cached_noise['sigma_'+dtype], data

    def _cache_observed_data(self, dtype, values, sigma, metadata, uvdist_sections,
                             n_baselines=1):
        """Cache unfiltered UV tracks and filtered values with augmented errors.

        CI values contain real components followed by imaginary components, so
        both halves use the same baseline mask.
        """
        if n_baselines == 1:
            uv = np.array([metadata['u'], metadata['v']])
        else:
            uv = np.array([[metadata[f'u{i}'], metadata[f'v{i}']]
                           for i in range(1, n_baselines + 1)])
        self.uv[dtype] = uv
        mask = self.filter_by_uvdist_ranges(uv, uvdist_sections)
        if dtype == 'ci':
            mask = np.concatenate([mask, mask])
        # Preserve augmentation order and array layout from each sampling path.
        # Masking a structured-array view first can change floating-point rounding.
        if dtype in ('camp', 'logcamp', 'ci'):
            values, sigma = values[mask], sigma[mask]
            sigma = utils.augment_sigma(values, sigma, self.noise_frac,
                                        self.static_noise_floor, self.noise_factor)
        else:
            sigma = utils.augment_sigma(values, sigma, self.noise_frac,
                                        self.static_noise_floor, self.noise_factor)[mask]
            values = values[mask]
        self.cached_data[dtype] = values
        self.cached_noise['sigma_'+dtype] = sigma

    def _ci_averaging_matrix(self, n_advariants, avg_time, scan_avg):
        return self._averaging_matrix(self._ci_raw_metadata, avg_time, scan_avg)

    def _averaging_matrix(self, metadata, avg_time, scan_avg):
        """Return weights in exactly the row order of coh_avg_ndarray."""
        n = len(metadata)
        if n == 0 or (avg_time <= 0 and not scan_avg):
            return np.eye(n)
        plan = self._coh_averaging_plan(metadata, avg_time, scan_avg)
        rows, times = [], []
        for group in plan['group_plans']:
            for j in range(group['n_bins']):
                indices = group['ord_idx'][group['bin_inv'] == j]
                weights = np.zeros(n)
                weights[indices] = 1 / len(indices)
                rows.append(weights)
                times.append(metadata['time'][indices].mean())
        return np.asarray(rows)[np.argsort(times)]

    def _cache_covariance(self, dtype, cov, features, augment=True):
        """Cache covariance, inverse, logdet and marginal errors in real features.

        Redundant closures can have singular covariance. Eigenvalues are floored
        at 1e-10 times the largest eigenvalue, as in the previous CI path.
        """
        cov = np.asarray(cov, dtype=float).copy()
        if augment:
            cov[np.diag_indices_from(cov)] += ((self.noise_frac * np.abs(features))**2
                                               + self.static_noise_floor**2)
            cov *= self.noise_factor**2
        if cov.size:
            if not np.all(np.isfinite(cov)):
                raise ValueError(f"Non-finite {dtype} covariance; check zero or low-SNR visibilities")
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            if eigenvalues[-1] <= 0:
                raise ValueError(f"{dtype} covariance has no positive variance")
            eigenvalues = np.maximum(eigenvalues, 1e-10 * eigenvalues[-1])
            cov = (eigenvectors * eigenvalues) @ eigenvectors.T
            inverse = (eigenvectors / eigenvalues) @ eigenvectors.T
            logdet = np.log(eigenvalues).sum()
        else:
            inverse, logdet = cov.copy(), 0.
        self.cached_noise['cov_'+dtype] = cov
        self.cached_noise['invcov_'+dtype] = inverse
        self.cached_noise['logcovdet_'+dtype] = logdet
        diagonal = np.diag(cov)
        if dtype in ('vis', 'bispec'):
            n = len(diagonal) // 2
            # sample_data retains one representative component error per complex datum.
            sigma = np.sqrt((diagonal[:n] + diagonal[n:]) / 2)
        else:
            sigma = np.sqrt(diagonal)
        self.cached_noise['sigma_'+dtype] = sigma

    def _cache_observable_covariance(self, dtype, min_sep_time, uvdist_sections):
        """Propagate visibility noise through the same operations as sample_data."""
        observed = self.cached_data[dtype]
        if dtype == 'vis':
            # Already averaged, filtered and augmented analytical component errors.
            variance = self.cached_noise['sigma_vis']**2
            self._cache_covariance(dtype, np.diag(np.r_[variance, variance]),
                                   utils.real_features(observed), augment=False)
            return
        N = max(self.cov_samples, 5 * observed.size * (2 if dtype == 'bispec' else 1))
        samples, metadata = self._draw_observable_samples(
            dtype, N, min_sep_time, visibility_metadata=self._reorder_obsdata_ndarray(self.obs)
            if dtype == 'visamp' else None)
        average = self._averaging_matrix(metadata, self.avg_time, self.scan_avg)
        if dtype == 'ci':
            n = len(metadata)
            samples = np.concatenate([samples[:, :n] @ average.T,
                                      samples[:, n:] @ average.T], axis=1)
        else:
            samples = samples @ average.T
            samples = self._transform_observable_samples(dtype, samples)
        mask = self.filter_by_uvdist_ranges(self.uv[dtype], uvdist_sections)
        if dtype == 'ci':
            mask = np.r_[mask, mask]
        samples = samples[:, mask]
        if dtype == 'cphase':
            samples = np.angle(np.exp(1j * (samples - observed)))
        if np.iscomplexobj(samples):
            samples = np.concatenate([samples.real, samples.imag], axis=1)
        cov = np.atleast_2d(np.cov(samples, rowvar=False)) if samples.shape[1] else np.empty((0, 0))
        features = utils.real_features(observed)
        if dtype == 'bispec':
            # Apply the same amplitude-based noise floor to each complex component.
            features = np.tile(np.abs(observed), 2)
        self._cache_covariance(dtype, cov, features)

    def _term_covariance(self, dtype, logcovdet, invcov):
        if self.use_cov:
            return (self.cached_noise['logcovdet_'+dtype],
                    self.cached_noise['invcov_'+dtype])
        return logcovdet, invcov

    def _covariance_term_loglike(self, dtype, data, model_data, logcovdet, invcov):
        if dtype == 'cphase':
            if utils.is_torch(model_data):
                residual = utils.to_torch(data, like=model_data) - model_data
                residual = torch.atan2(torch.sin(residual), torch.cos(residual))
            else:
                residual = np.angle(np.exp(1j * (data - model_data)))
            return self._covariance_loglike(residual, residual * 0, logcovdet, invcov)
        return self._covariance_loglike(utils.real_features(data),
                                       utils.real_features(model_data), logcovdet, invcov)

    def _log_likelihood_dtype(self, parameters, dtype, logcovdet=None, invcov=None):
        """Evaluate one term, sampling before retrieving cached covariance.

        Explicit logcovdet/invcov are used when use_cov is False. Complex
        covariance features and phase wrapping are handled by the covariance
        helper; the diagonal closure-phase path keeps its cosine residual.
        """
        data, sigma, model_data = self.sample_data(parameters, dtype=dtype, ttype='analytical')
        logcovdet, invcov = self._term_covariance(dtype, logcovdet, invcov)
        if dtype != 'cphase':
            if logcovdet is not None and invcov is not None:
                return self._covariance_term_loglike(dtype, data, model_data, logcovdet, invcov)
            return self._gaussian_loglike(data, model_data, sigma)
        if utils.is_torch(model_data):
            data_t = utils.to_torch(data, like=model_data)
            sigma_t = utils.to_torch(sigma, like=model_data)
            residual = 1 - torch.cos(data_t - model_data)
            if logcovdet is not None and invcov is not None:
                return self._covariance_term_loglike('cphase', data_t, model_data, logcovdet, invcov)
            return -0.5 * torch.sum(residual / sigma_t ** 2 + torch.log(2 * torch.pi * sigma_t ** 2))
        residual = 1 - np.cos(data - model_data)
        if logcovdet is not None and invcov is not None:
            return self._covariance_term_loglike('cphase', data, model_data, logcovdet, invcov)
        return -0.5 * np.sum(residual / sigma ** 2 + np.log(2 * np.pi * sigma ** 2))

    def log_likelihood_vis(self, parameters, logcovdet=None, invcov=None):
        """Evaluate the visibility likelihood; see _log_likelihood_dtype."""
        return self._log_likelihood_dtype(parameters, 'vis', logcovdet, invcov)

    def log_likelihood_visamp(self, parameters, logcovdet=None, invcov=None):
        """Evaluate the visibility amplitude likelihood; see _log_likelihood_dtype."""
        return self._log_likelihood_dtype(parameters, 'visamp', logcovdet, invcov)

    def log_likelihood_bispec(self, parameters, logcovdet=None, invcov=None):
        """Evaluate the bispectrum likelihood; see _log_likelihood_dtype."""
        return self._log_likelihood_dtype(parameters, 'bispec', logcovdet, invcov)

    def log_likelihood_cphase(self, parameters, logcovdet=None, invcov=None):
        """Evaluate the closure phase likelihood; see _log_likelihood_dtype."""
        return self._log_likelihood_dtype(parameters, 'cphase', logcovdet, invcov)

    def log_likelihood_camp(self, parameters, logcovdet=None, invcov=None):
        """Evaluate the closure amplitude likelihood; see _log_likelihood_dtype."""
        return self._log_likelihood_dtype(parameters, 'camp', logcovdet, invcov)

    def log_likelihood_logcamp(self, parameters, logcovdet=None, invcov=None):
        """Evaluate the log closure amplitude likelihood; see _log_likelihood_dtype."""
        return self._log_likelihood_dtype(parameters, 'logcamp', logcovdet, invcov)

    def log_likelihood_ci(self, parameters, logcovdet=None, invcov=None):
        """Evaluate the closure invariant likelihood; see _log_likelihood_dtype."""
        return self._log_likelihood_dtype(parameters, 'ci', logcovdet, invcov)

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
            Component standard deviations packed as sigma_real + 1j * sigma_imag.
            They are equal under the circular visibility-noise approximation.
        metadata : np.ndarray
            Structured metadata for each advariant sample.
        """
        return self._calc_triangle_products(N, add_noise, count, min_sep_time, 'adv')

    def calc_bispectrum(self, N=1, add_noise=False, count='min', min_sep_time=0):
        """Return triangle visibility products, component errors and metadata.

        Metadata contains time, stations, directed UV coordinates, bispec and
        sigma_bispec. sigma_bispec is the first-order standard deviation of either complex
        component. Noise samples share draws for repeated physical baselines.
        """
        return self._calc_triangle_products(N, add_noise, count, min_sep_time, 'bispec')

    @staticmethod
    def sample_visibility_noise(visibilities, sigmas, baseline_keys, N):
        """Draw circular Gaussian visibility noise with shared baseline errors.

        Each input sequence contains one array per baseline position in an
        observable. Keys are (physical_baseline_id, conjugated) pairs. Repeated
        IDs reuse a draw, conjugated when required. Results have shape (N, M).
        This sampler can also be used with quadrangles and other observables.
        """
        if not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError("N must be a positive integer")
        draws = {}
        samples = []
        for values, errors, keys in zip(visibilities, sigmas, baseline_keys):
            columns = []
            for (key, conjugated), sigma in zip(keys, errors):
                if key not in draws:
                    draws[key] = sigma * (np.random.normal(size=N) + 1j * np.random.normal(size=N))
                columns.append(draws[key].conj() if conjugated else draws[key])
            noise = np.column_stack(columns) if columns else np.empty((N, 0), complex)
            samples.append(np.asarray(values)[None, :] + noise)
        return samples

    def _visibility_time_groups(self):
        """Group obs.data by time and include conjugate directed baselines."""
        # Construct directed baselines from the visibility table alone.
        raw = self.obs.data
        tlist = []
        for time in np.unique(raw['time']):
            rows = raw[raw['time'] == time]
            reverse = rows.copy()
            reverse['t1'], reverse['t2'] = rows['t2'], rows['t1']
            reverse['u'], reverse['v'] = -rows['u'], -rows['v']
            reverse['vis'] = rows['vis'].conj()
            tlist.append(np.concatenate([rows, reverse]))
        return tlist

    def calc_camp(self, N=1, add_noise=False, count='min', min_sep_time=0):
        """Calculate un-debiased closure amplitudes from visibility quadrangles.

        Baseline order is (12), (34), (14), (23), giving |V12 V34/V14 V23|.
        Return values, first-order standard deviations and metadata containing
        time, t1..t4, u1/v1..u4/v4, camp and sigma_camp. Noisy outputs retain
        shape (N, quadrangles); deterministic outputs have shape (quadrangles,).
        Shared visibility draws preserve correlations between quadrangles.
        """
        if count not in ('min', 'max'):
            raise ValueError("count must be 'min' or 'max'")
        if not isinstance(N, (int, np.integer)) or N < 1:
            raise ValueError("N must be a positive integer")
        N = N if add_noise else 1
        metadata_dtype = np.dtype(
            [('time', '<f8')] + [(f't{i}', '<U32') for i in range(1, 5)] +
            [(name, '<f8') for i in range(1, 5) for name in (f'u{i}', f'v{i}')] +
            [('camp', '<f8'), ('sigma_camp', '<f8')])
        entries, baselines = [], []
        last_time = {}
        for rows in self._visibility_time_groups():
            sites = sorted(set(rows['t1']) | set(rows['t2']))
            if len(sites) < 4:
                continue
            lookup = {(row['t1'], row['t2']): row for row in rows}
            if count == 'min':
                quads = utils.quad_minimal_set(sites, self.obs.tarr, self.obs.tkey)
            else:
                quads = [quad for a, b, c, d in it.combinations(sites, 4)
                         for quad in ((a, b, c, d), (a, c, b, d), (a, b, d, c))]
            for quad in quads:
                quad = tuple(quad)
                a, b, c, d = quad
                try:
                    selected = [lookup[pair] for pair in ((a, b), (c, d), (a, d), (b, c))]
                except KeyError:
                    continue
                time = float(rows['time'][0])
                if quad in last_time and time - last_time[quad] < min_sep_time / 3600:
                    continue
                last_time[quad] = time
                entries.append((time, *quad, *[value for row in selected for value in (row['u'], row['v'])]))
                baselines.append(selected)
        if not baselines:
            empty = np.empty((N, 0)) if add_noise else np.empty(0)
            return empty, empty.copy(), np.empty(0, dtype=metadata_dtype)
        vis = [np.array([rows[i]['vis'] for rows in baselines]) for i in range(4)]
        sigmas = [np.array([rows[i]['sigma'] for rows in baselines]) for i in range(4)]
        if add_noise:
            keys = [[((float(row[i]['time']), *sorted((str(row[i]['t1']), str(row[i]['t2'])))),
                      str(row[i]['t1']) > str(row[i]['t2'])) for row in baselines]
                    for i in range(4)]
            vis = self.sample_visibility_noise(vis, sigmas, keys, N)
        a, b, c, d = [np.abs(v) for v in vis]
        with np.errstate(divide='ignore', invalid='ignore'):
            values = a * b / (c * d)
            # Derivative form avoids 0/0 for zero numerator amplitudes.
            errors = np.sqrt((sigmas[0] * b / (c*d))**2 +
                             (sigmas[1] * a / (c*d))**2 +
                             (sigmas[2] * values / c)**2 +
                             (sigmas[3] * values / d)**2)
        first_values = values[0] if add_noise else values
        first_errors = errors[0] if add_noise else errors
        metadata = np.array([(*entry, value, error) for entry, value, error
                             in zip(entries, first_values, first_errors)], dtype=metadata_dtype)
        return values, errors, metadata

    def _calc_triangle_products(self, N, add_noise, count, min_sep_time, product):
        tlist = self._visibility_time_groups()
        if not add_noise:
            N = 1

        metadata_dtype = np.dtype([
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
        ])
        if product == 'bispec':
            metadata_dtype = np.dtype(metadata_dtype.descr +
                                          [('bispec', '<c16'), ('sigma_bispec', '<f8')])
        elif product == 'adv':
            metadata_dtype = np.dtype(metadata_dtype.descr +
                                            [('adv', '<c16'), ('sigma_adv', '<c16'), ('ci', '<c16'), ('sigma_ci', '<c16')])

        metadata_rows = []
        triad_time_dict = {}

        v1s, v2s, v3s = [], [], []
        s1s, s2s, s3s = [], [], []
        metadata_entries = []
        baseline_keys = [[], [], []]

        for tdata in tlist:
            tdata = tdata.astype(self.obs.data.dtype)
            sites = sorted(set(np.hstack((tdata['t1'], tdata['t2']))))

            # Create a dictionary of baselines at the current time incl. conjugates;
            l_dict = {}
            for dat in tdata:
                l_dict[(dat['t1'], dat['t2'])] = dat

            if count == 'max':
                tris = np.sort(list(it.combinations(sites, 3)))
            else:
                tris = utils.tri_minimal_set(sites, self.obs.tarr, self.obs.tkey)

            for tri in tris:
                tri = tuple(tri)
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

                for keys, baseline in zip(baseline_keys, (l1, l2, l3)):
                    station1, station2 = str(baseline['t1']), str(baseline['t2'])
                    keys.append(((float(baseline['time']), *sorted((station1, station2))),
                                 station1 > station2))
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
            advariants_noise = np.empty((N, 0), dtype=np.complex128)
            metadata = np.array(metadata_rows, dtype=metadata_dtype)
            return (advariants if add_noise else advariants[0],
                    advariants_noise.real if product == 'bispec' and add_noise else
                    advariants_noise[0].real if product == 'bispec' else
                    advariants_noise if add_noise else advariants_noise[0], metadata)

        v1, v2, v3 = [np.asarray(x, dtype=np.complex128) for x in (v1s, v2s, v3s)]
        s1, s2, s3 = [np.asarray(x, dtype=np.float64) for x in (s1s, s2s, s3s)]

        if add_noise:
            v1, v2, v3 = self.sample_visibility_noise(
                (v1, v2, v3), (s1, s2, s3), baseline_keys, N)

        adv = v1 * v2 * v3 if product == 'bispec' else v1 * np.conj(v2)**(-1) * v3
        if product == 'bispec':
            adv_noise = np.sqrt((s1 * np.abs(v2 * v3))**2
                                + (s2 * np.abs(v1 * v3))**2
                                + (s3 * np.abs(v1 * v2))**2)
            adv_noise = np.atleast_2d(adv_noise)
        else:
            relative_variance = ((s1[None, :] / np.abs(v1))**2
                                 + (s2[None, :] / np.abs(v2))**2
                                 + (s3[None, :] / np.abs(v3))**2)
            adv_noise = np.sqrt(relative_variance) * np.abs(adv)

        advariants = np.asarray(adv, dtype=np.complex128)
        advariants_noise = np.asarray(adv_noise * (1 + 1j), dtype=np.complex128)
        adv0 = advariants[0] if advariants.ndim > 1 else advariants
        adv_noise0 = advariants_noise[0] if advariants_noise.ndim > 1 else advariants_noise
        advariants = advariants.reshape(N, -1) if add_noise else advariants.reshape(-1)
        advariants_noise = advariants_noise.reshape(N, -1) if add_noise else advariants_noise.reshape(-1)

        if product == 'bispec':
            metadata_rows = [(*entry, adv0[idx], adv_noise0[idx].real)
                             for idx, entry in enumerate(metadata_entries)]
            advariants_noise = advariants_noise.real
        else:
            metadata_rows = [(*entry, adv0[idx], adv_noise0[idx], np.nan, np.nan)
                             for idx, entry in enumerate(metadata_entries)]
        metadata = np.array(metadata_rows, dtype=metadata_dtype)

        return advariants, advariants_noise, metadata
    
    def advariants_to_ci(self, advariants, times, advariants_noise=None):
        """Normalise advariants to closure invariants using uncertainty propagation.
        
        Uses analytical first-order propagation, assuming independent Gaussian
        real and imaginary advariant components. Cross-advariant correlations
        are ignored.

        Parameters
        ----------
        advariants : np.ndarray
            The advariants to normalise. May be a single 1D array of length M or a
            stacked 2D array of shape (N, M) for multiple noise realisations.
        times : np.ndarray
            The times corresponding to the advariants. Must be 1D with length M.
        advariants_noise : np.ndarray, optional
            Component standard deviations packed as sigma_real + 1j * sigma_imag.
            Real inputs apply the same sigma to both components.
            Must match the shape of advariants.

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

        groups, order, starts = self._ci_time_groups(times)
        values = np.concatenate([advariants.real, advariants.imag], axis=1)
        if times.size == 0:
            ci = values[0] if squeeze_output else values
            return ci, np.empty_like(ci) if advariants_noise is not None else None
        sum_norm_p = np.add.reduceat(np.abs(values[:, order])**self.normpower, starts, axis=1)
        norm = sum_norm_p[:, groups]**(1 / self.normpower)
        ci = values / norm

        sigma_ci = None
        if advariants_noise is not None:
            # Legacy real inputs apply the same sigma to both components.
            sigma_real = advariants_noise.real
            sigma_imag = advariants_noise.imag if np.iscomplexobj(advariants_noise) else sigma_real
            sigma = np.concatenate([sigma_real, sigma_imag], axis=1)
            dnorm_dx = np.abs(values)**(self.normpower - 1) * np.sign(values) / norm**(self.normpower - 1)
            norm_variance = np.add.reduceat((dnorm_dx * sigma)[:, order]**2, starts, axis=1)
            variance = ((sigma / norm)**2 + values**2 * norm_variance[:, groups] / norm**4
                        - 2 * values * sigma**2 * dnorm_dx / norm**3)
            sigma_ci = np.sqrt(np.maximum(variance, 0))

        if squeeze_output:
            return ci[0], sigma_ci[0] if sigma_ci is not None else None
        return ci, sigma_ci

    def _ci_time_groups(self, times):
        """Cache the grouping used by real and imaginary CI components."""
        cached = getattr(self, '_ci_group_cache', None)
        if cached is None or not np.array_equal(cached[0], times):
            _, groups = np.unique(times, return_inverse=True)
            groups = np.concatenate([groups, groups])
            order = np.argsort(groups, kind='stable')
            starts = np.flatnonzero(np.r_[True, np.diff(groups[order]) != 0])
            cached = (times.copy(), groups, order, starts)
            self._ci_group_cache = cached
        return cached[1:]

    def _draw_observable_samples(self, dtype, N, min_sep_time, visibility_metadata=None):
        """Return MC samples and metadata before averaging and final transforms.

        CI is normalized at each original timestamp. Visibility amplitudes,
        closure phases and log closure amplitudes are transformed by the caller
        after any averaging, matching the corresponding observed-data path.
        """
        if dtype == 'visamp':
            metadata = self.obs.data if visibility_metadata is None else visibility_metadata
            keys = [((float(row['time']), *sorted((str(row['t1']), str(row['t2'])))),
                     str(row['t1']) > str(row['t2'])) for row in metadata]
            samples = self.sample_visibility_noise(
                [metadata['vis']], [metadata['sigma']], [keys], N)[0]
        else:
            if dtype in ('bispec', 'cphase'):
                calculate = self.calc_bispectrum
            elif dtype in ('camp', 'logcamp'):
                calculate = self.calc_camp
            elif dtype == 'ci':
                calculate = self.calc_advariants
            else:
                raise ValueError("Unsupported dtype: " + str(dtype))
            samples, _, metadata = calculate(
                N=N, add_noise=True, count=self.count, min_sep_time=min_sep_time)
            if dtype == 'ci':
                samples, _ = self.advariants_to_ci(samples, metadata['time'])
        return samples, metadata

    @staticmethod
    def _transform_observable_samples(dtype, samples):
        """Apply the final observable transform after any averaging."""
        if dtype == 'visamp':
            return np.abs(samples)
        if dtype == 'cphase':
            return np.angle(samples)
        if dtype == 'logcamp':
            return np.log(samples)
        return samples

    def estimate_noise(self, dtype='ci', N=5000, calc_cov=True, min_sep_time=0):
        """Estimate raw observable errors and covariance from visibility noise.

        Visibility and bispectrum features are [real, imaginary]; closure phases are radians,
        wrapped about the observed phase to avoid the +/-pi discontinuity.
        These samples precede averaging, UV filtering and noise augmentation.
        Other observables can reuse sample_visibility_noise and utils.estimate_mc_noise.
        """
        if dtype == 'vis':
            variance = np.tile(self.obs.data['sigma']**2, 2)
            return np.sqrt(variance), np.diag(variance) if calc_cov else None
        samples, _ = self._draw_observable_samples(dtype, N, min_sep_time)
        samples = self._transform_observable_samples(dtype, samples)
        if dtype == 'cphase':
            observed, _, _ = self.calc_bispectrum(count=self.count, min_sep_time=min_sep_time)
            samples = np.angle(np.exp(1j * (samples - np.angle(observed))))
        return utils.estimate_mc_noise(samples, calc_cov=calc_cov)

    def estimate_noise_ci(self, N=5000, calc_cov=True):
        """Backward-compatible wrapper for closure invariant Monte Carlo errors."""
        return self.estimate_noise('ci', N=N, calc_cov=calc_cov)

    def filter_by_uvdist_ranges(self, uv, uv_ranges):
        if uv_ranges is None:
            return np.ones(np.asarray(uv).shape[-1], dtype=bool)
        uv = np.asarray(uv)
        ranges = tuple(tuple(bounds) for bounds in uv_ranges)
        if not hasattr(self, '_uv_mask_cache'):
            self._uv_mask_cache = {}
        key = (uv.shape, uv.dtype.str, ranges)
        cached = self._uv_mask_cache.get(key)
        if cached is None or not np.array_equal(cached[0], uv):
            uvdists = utils.calc_uvdist(uv)
            mask = np.zeros_like(uvdists, dtype=bool)
            for uv_min, uv_max in ranges:
                mask |= (uvdists >= uv_min) & (uvdists <= uv_max)
            cached = (uv.copy(), mask)
            self._uv_mask_cache[key] = cached
        return cached[1].copy()

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

        self.clear_cached_data()

        if avg_time is None:
            avg_time = self.avg_time
        if scan_avg is None:
            scan_avg = self.scan_avg
        cached_uvdist_sections = self.uvdist_sections
        self.uvdist_sections = None

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
            data, sigma, model_data = self.sample_data(parameters, dtype=dtype, ttype='analytical', avg_time=avg_time, scan_avg=scan_avg, uvdist_sections=None)
            uvdist = utils.calc_uvdist(self.uv[dtype])

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

        self.clear_cached_data()
        self.uvdist_sections = cached_uvdist_sections

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
            ci_err = np.concatenate([ci_err.real, ci_err.imag])
            return ci, ci_err
        else:
            complex_ci = ci[:len(ci)//2] + 1j*ci[len(ci)//2:]
            ci_err = ci_err[:len(ci_err)//2] + 1j * ci_err[len(ci_err)//2:]
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

    def _coh_averaging_plan(self, data_array, avg_time, scan_avg, dt=0.0165, margin=0.0001):
        """Build or reuse the grouping shared by table and matrix averaging."""
        dtype = data_array.dtype
        names = dtype.names

        tn_cols = [c for c in names if c.startswith('t') and c != 'time']
        sigma_cols = [c for c in names if 'sigma' in c]
        mean_cols = [c for c in names if c not in tn_cols + sigma_cols + ['time']]

        if not hasattr(self, "_coh_avg_plan_cache"):
            self._coh_avg_plan_cache = {}
            self._coh_avg_plan_cache_base = None

        t = data_array['time']
        base_sig = (len(data_array), dtype.descr, tuple(tn_cols), t.dtype.str,
                    float(t[0]) if len(t) else None, float(t[-1]) if len(t) else None)

        if self._coh_avg_plan_cache_base != base_sig:
            self._coh_avg_plan_cache.clear()
            self._coh_avg_plan_cache_base = base_sig

        plan_key = (float(avg_time), bool(scan_avg), float(dt), float(margin))
        plan = self._coh_avg_plan_cache.get(plan_key)

        if plan is not None:
            return plan

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
            elif avg_time == 0:
                _, bin_inv = np.unique(tg, return_inverse=True)
            else:
                # Each bin starts at its first sample, rather than a fixed grid.
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

            group_plans.append(dict(ord_idx=ord_idx, t_sorted=tg, bin_inv=bin_inv,
                                    counts=counts, n_bins=n_bins, tn_ref_idx=ord_idx[0]))

        plan = dict(group_plans=group_plans, tn_cols=tn_cols, mean_cols=mean_cols,
                    sigma_cols=sigma_cols, dtype=dtype)
        self._coh_avg_plan_cache[plan_key] = plan

        return plan

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

        if len(data_array) == 0 or (avg_time <= 0 and not scan_avg):
            return data_array

        plan = self._coh_averaging_plan(data_array, avg_time, scan_avg, dt, margin)
        dtype = data_array.dtype
        tn_cols, mean_cols, sigma_cols = (plan[key] for key in ('tn_cols', 'mean_cols', 'sigma_cols'))

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
                    chunk[c] = np.bincount(bin_inv, weights=vals.astype(np.float64, copy=False),
                                           minlength=n_bins) / counts

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
