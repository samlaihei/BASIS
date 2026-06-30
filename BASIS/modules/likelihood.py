from BASIS.models import base

import bilby
import numpy as np
import torch
import itertools as it

class ModelLikelihood(bilby.Likelihood):
    def __init__(self, model_names, obs, img=None, imgdim=128, fov=225, count='min',  dterms={'vis':100},
                 static_noise_floor=0, noise_frac=0, noise_factor=1):
        """Initialize the ModelLikelihood class.

        Parameters
        ----------
        model_names : list of str
            The names of the models to use.
        obs : eh.obsdata.ObsData
            The observational data.
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

    def sample_data(self, parameters, dtype='vis', ttype='analytical'):
        """Sample the model data for the given parameters and data type.
        
        Parameters
        ----------
        parameters : dict
            The model parameters.
        dtype : str, optional
            The data type to sample ('image', 'vis', 'visamp', 'bispec', 'cphase', 'camp', 'logcamp', 'ci'). Default is 'vis'.
        ttype : str, optional
            The type of sampling ('analytical' or 'dft'). Default is 'analytical'.

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
        
        data = None
    
        if dtype == 'image' and self.img is not None:
            if dtype not in self.cached_data:
                self.cached_data[dtype] = self.img
                self.cached_noise['sigma'+dtype] = np.ones_like(self.img._imdict['I'])*1e-4  # Assuming uniform noise for image
            if parameters is not None:
                data = model.sky_map()
        
        elif dtype == 'vis':
            if dtype not in self.cached_data:
                self.cached_data[dtype] = self.obs.data['vis']
                self.cached_noise['sigma'+dtype] = self.obs.data['sigma']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([self.obs.data['u'], self.obs.data['v']])
            if parameters is not None:
                data = model.sample_vis(self.uv[dtype], ttype=ttype)
            
        elif dtype == 'visamp':
            if dtype not in self.cached_data:
                self.cached_data[dtype] = np.abs(self.obs.data['vis'])
                self.cached_noise['sigma'+dtype] = self.obs.data['sigma']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([self.obs.data['u'], self.obs.data['v']])
            if parameters is not None:
                vis_data = model.sample_vis(self.uv[dtype], ttype=ttype)
                data = torch.abs(vis_data) if self._is_torch(vis_data) else np.abs(vis_data)
        
        elif dtype == 'bispec':
            if dtype not in self.cached_data:
                self.cached_data[dtype] = self.obs.bispec['bispec']
                self.cached_noise['sigma'+dtype] = self.obs.bispec['sigmab']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[self.obs.bispec['u1'], self.obs.bispec['v1']],
                                [self.obs.bispec['u2'], self.obs.bispec['v2']],
                                [self.obs.bispec['u3'], self.obs.bispec['v3']]])
            if parameters is not None:
                vis1 = model.sample_vis(self.uv[dtype][0, :], ttype=ttype)
                vis2 = model.sample_vis(self.uv[dtype][1, :], ttype=ttype)
                vis3 = model.sample_vis(self.uv[dtype][2, :], ttype=ttype)
                data = vis1 * vis2 * vis3
        
        elif dtype == 'cphase':
            if dtype not in self.cached_data:
                self.cached_data[dtype] = self.obs.cphase['cphase']*np.pi/180
                self.cached_noise['sigma'+dtype] = self.obs.cphase['sigmacp']*np.pi/180
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[self.obs.cphase['u1'], self.obs.cphase['v1']],
                                [self.obs.cphase['u2'], self.obs.cphase['v2']],
                                [self.obs.cphase['u3'], self.obs.cphase['v3']]])
            if parameters is not None:
                vis1 = model.sample_vis(self.uv[dtype][0, :], ttype=ttype)
                vis2 = model.sample_vis(self.uv[dtype][1, :], ttype=ttype)
                vis3 = model.sample_vis(self.uv[dtype][2, :], ttype=ttype)
                bispec = vis1 * vis2 * vis3
                data = torch.angle(bispec) if self._is_torch(bispec) else np.angle(bispec)
        
        elif dtype == 'camp':
            if dtype not in self.cached_data:
                self.cached_data[dtype] = self.obs.camp['camp']
                self.cached_noise['sigma'+dtype] = self.obs.camp['sigmaca']
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[self.obs.camp['u1'], self.obs.camp['v1']],
                                [self.obs.camp['u2'], self.obs.camp['v2']],
                                [self.obs.camp['u3'], self.obs.camp['v3']],
                                [self.obs.camp['u4'], self.obs.camp['v4']]])
            if parameters is not None:
                vis1 = model.sample_vis(self.uv[dtype][0, :], ttype=ttype)
                vis2 = model.sample_vis(self.uv[dtype][1, :], ttype=ttype)
                vis3 = model.sample_vis(self.uv[dtype][2, :], ttype=ttype)
                vis4 = model.sample_vis(self.uv[dtype][3, :], ttype=ttype)
                data = vis1 * vis2 / (vis3 * vis4)
        
        elif dtype == 'logcamp':
            if dtype not in self.cached_data:
                self.cached_data[dtype] = np.log(self.obs.camp['camp'])
                self.cached_noise['sigma'+dtype] = self.obs.camp['sigmaca']/np.abs(self.obs.camp['camp'])
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[self.obs.camp['u1'], self.obs.camp['v1']],
                                [self.obs.camp['u2'], self.obs.camp['v2']],
                                [self.obs.camp['u3'], self.obs.camp['v3']],
                                [self.obs.camp['u4'], self.obs.camp['v4']]])
            if parameters is not None:
                vis1 = model.sample_vis(self.uv[dtype][0, :], ttype=ttype)
                vis2 = model.sample_vis(self.uv[dtype][1, :], ttype=ttype)
                vis3 = model.sample_vis(self.uv[dtype][2, :], ttype=ttype)
                vis4 = model.sample_vis(self.uv[dtype][3, :], ttype=ttype)
                if any(self._is_torch(v) for v in [vis1, vis2, vis3, vis4]):
                    data = torch.log(vis1) + torch.log(vis2) - torch.log(vis3) - torch.log(vis4)
                else:
                    data = np.log(vis1) + np.log(vis2) - np.log(vis3) - np.log(vis4)
        
        elif dtype == 'ci':
            if dtype not in self.cached_data:
                adv, adv_noise = self.calc_advariants(count=self.count)
                self.cached_data[dtype], self.cached_noise['sigma'+dtype] = self.advariants_to_ci(adv, self.obs.bispec['time'], adv_noise)
                self.cached_noise['sigma'+dtype] = self.estimate_noise_ci() # MC-style estimation of CI noise, assuming independent Gaussian noise on visibilities
                self.cached_noise['sigma'+dtype] = self.augment_sigma(self.cached_data[dtype], self.cached_noise['sigma'+dtype])
                self.uv[dtype] = np.array([[self.obs.bispec['u1'], self.obs.bispec['v1']],
                                [self.obs.bispec['u2'], self.obs.bispec['v2']],
                                [self.obs.bispec['u3'], self.obs.bispec['v3']]])
            if parameters is not None:
                vis1 = model.sample_vis(self.uv[dtype][0, :], ttype=ttype)
                vis2 = model.sample_vis(self.uv[dtype][1, :], ttype=ttype)
                vis3 = model.sample_vis(self.uv[dtype][2, :], ttype=ttype)
                if any(self._is_torch(v) for v in [vis1, vis2, vis3]):
                    advariants = vis1 * torch.conj(vis2).pow(-1) * vis3
                    data = self.advariants_to_ci_torch(advariants, self.obs.bispec['time'])
                else:
                    advariants = vis1 * np.conj(vis2)**(-1) * vis3
                    data = self.advariants_to_ci(advariants, self.obs.bispec['time'])[0]

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

    def calc_advariants(self, N=1, add_noise=False, count='min'):
        """Calculate the advariants from the observational data.

        Parameters
        ----------
        N : int, optional
            The number of advariant samples to calculate (default is 1).
        add_noise : bool, optional
            Whether to add noise to the visibilities (default is False).
        count : str, optional
            'min' to use minimum count of triangles, 'max' to use maximum (default is 'min').

        Returns
        -------
        advariants : np.ndarray
            The calculated advariants.
        """
        tlist = self.obs.tlist(conj=True)
        if not add_noise:
            N = 1
        advariants = [[] for i in range(N)]
        advariants_noise = [[] for i in range(N)]
        for tdata in tlist:
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
                for i in range(N):
                    try:
                        # Select triangle entries in the data dictionary
                        l1 = l_dict[(tri[0], tri[1])]
                        l2 = l_dict[(tri[1], tri[2])]
                        l3 = l_dict[(tri[2], tri[0])]
                    except KeyError:
                        # If any baseline is missing, skip this triangle
                        continue

                    v1, v2, v3 = l1['vis'], l2['vis'], l3['vis']
                    s1, s2, s3 = l1['sigma'], l2['sigma'], l3['sigma']
                    
                    if add_noise:
                        v1 += (np.random.normal(0, s1) + 1j * np.random.normal(0, s1))
                        v2 += (np.random.normal(0, s2) + 1j * np.random.normal(0, s2))
                        v3 += (np.random.normal(0, s3) + 1j * np.random.normal(0, s3))

                    adv = v1 * np.conj(v2)**(-1) * v3
                    advariants[i].append(adv)
                    advariants_noise[i].append(np.sqrt((s1/np.abs(v1))**2 + (s2/np.abs(v2))**2 + (s3/np.abs(v3))**2) * np.abs(adv))

        advariants = np.squeeze(np.array(advariants).reshape(N, -1))
        advariants_noise = np.squeeze(np.array(advariants_noise).reshape(N, -1))/np.sqrt(2)
        return advariants, advariants_noise
    
    def advariants_to_ci(self, advariants, times, advariants_noise=None):
        """Normalise advariants to closure invariants using uncertainty propagation.
        
        Uses the uncertainties package for automatic error propagation if available,
        otherwise falls back to analytical error propagation. Assumes independent Gaussian noise on the advariants.

        Parameters
        ----------
        advariants : np.ndarray
            The advariants to normalise.
        times : np.ndarray
            The times corresponding to the advariants.
        advariants_noise : np.ndarray, optional
            The noise associated with the advariants.

        Returns
        -------
        real_valued_advariants : np.ndarray
            The normalised closure invariants.
        real_valued_advariants_noise : np.ndarray
            The normalised closure invariants noise.
        """
        unique_times = np.unique(times)
        time_axis = np.concatenate([times, times], axis=0)
        real_valued_advariants = np.concatenate([np.real(advariants), np.imag(advariants)], axis=0)
        if advariants_noise is not None:
            real_valued_advariants_noise = np.concatenate([advariants_noise, advariants_noise], axis=0)
        
        norms_dict = {}
        norms_noise_dict = {}

        for t in unique_times:
            mask = (time_axis == t)
            values = real_valued_advariants[mask]
            
            sum_norm_p = np.sum(np.abs(values)**self.normpower)
            norm = sum_norm_p**(1/self.normpower)
            norms_dict[t] = norm
            
            if advariants_noise is not None:
                sigmas = real_valued_advariants_noise[mask]
                dnorm_dx = (np.abs(values)**(self.normpower-1) * np.sign(values) / 
                            sum_norm_p**(1 - 1/self.normpower))
                
                norm_variance = np.sum((dnorm_dx * sigmas)**2)
                norms_noise_dict[t] = np.sqrt(norm_variance)

        for i, t in enumerate(time_axis):
            norm = norms_dict[t]
            value_unnormalized = real_valued_advariants[i]
            real_valued_advariants[i] /= norm
            
            if advariants_noise is not None:
                sigma_x = real_valued_advariants_noise[i]
                sigma_norm = norms_noise_dict[t]
                variance = (sigma_x / norm)**2 + (value_unnormalized * sigma_norm / norm**2)**2
                real_valued_advariants_noise[i] = np.sqrt(variance)

        if advariants_noise is not None:
            return real_valued_advariants, real_valued_advariants_noise
        
        return real_valued_advariants, None
    
    
    def estimate_noise_ci(self, N=1000): # MC-style, assuming Gaussian noise on visibilities and propagating to closure invariants
        """Estimate the noise for closure invariant data.

        Returns
        -------
        sigma_ci : np.ndarray
            The estimated noise for closure invariant data.
        """

        advariants_noisy = self.calc_advariants(N=N, add_noise=True, count=self.count)[0]

        ci_noisy_samples = []
        for i in range(N):
            ci_noisy = self.advariants_to_ci(advariants_noisy[i], self.obs.bispec['time'])[0]
            ci_noisy_samples.append(ci_noisy)
        ci_noisy_samples = np.array(ci_noisy_samples)

        sigma_ci = np.std(ci_noisy_samples, axis=0)
        return sigma_ci

    def plot_all(self, parameters, save_path=None):
        """Plot the data and model for all data types.

        Parameters
        ----------
        parameters : dict
            The model parameters.
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 4, figsize=(16,12))
        errorbar_alpha = 0.4

        dterms = {'Visibility Amplitude': 'visamp', 'Visibility Phase': 'vis', 'Bispectrum Amplitude': 'bispec',
                'Closure Phase': 'cphase', 'Closure Amplitude': 'camp', 'Log Closure Amplitude': 'logcamp', 'Closure Invariants Amplitude': 'ci', 'Closure Invariants Phase': 'ci'}
        ops = {'Visibility Amplitude': np.abs, 'Visibility Phase': np.angle, 'Bispectrum Amplitude': np.abs, 
            'Closure Phase': lambda x: x, 'Closure Amplitude': np.abs, 'Log Closure Amplitude': lambda x: x, 'Closure Invariants Amplitude': np.abs, 'Closure Invariants Phase': np.angle}

        for i, (title, dtype) in enumerate(dterms.items()):
            data, sigma, model_data = self.sample_data(parameters, dtype=dtype, ttype='analytical')
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
                
            axs[i%2, i//2].errorbar(uvdist, ops[title](data), yerr=sigma, fmt='o', color='blue', markersize=0, capsize=1, alpha=errorbar_alpha, label='Data')
            if parameters is not None:
                axs[i%2, i//2].scatter(uvdist, ops[title](model_data), color='red', s=5, label='Model')
            axs[i%2, i//2].set_title(title)
            if i == 0:
                axs[i%2, i//2].legend()

        fig.text(0.5, 0.04, 'Total uv-Distance (λ)', ha='center', va='center', fontsize=18)
        fig.text(0.08, 0.5, 'Observable', ha='center', va='center', rotation='vertical', fontsize=18)

        if save_path is not None:
            plt.savefig(save_path, dpi=200)

        return fig, axs
        
    def sample_all(self, parameters, ttype='analytical'):
        """Sample the model data for all data types.

        Parameters
        ----------
        parameters : dict
            The model parameters.
        ttype : str, optional
            The type of sampling ('analytical' or 'dft'). Default is 'analytical'.

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
            data, sigma, model_data = self.sample_data(parameters, dtype=dtype, ttype=ttype)
            all_data[dtype] = data
            all_sigma[dtype] = sigma
            all_model_data[dtype] = model_data
        return all_data, all_sigma, all_model_data