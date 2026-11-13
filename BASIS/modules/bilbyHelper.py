"""Reusable Bilby fitting setup for scripts and notebooks.

Example
-------
>>> fit = BilbyHelper('data.uvfits', ['xsring', 'gauss'], {'cphase': 1, 'logcamp': 1},
...                   label='ring', ncpu=16, npoints=1000,
...                   sampler_kwargs={'sample': 'acceptance-walk', 'naccept': 30})
>>> result = fit.run()
>>> fit.plot_results(result)

Inspect or edit ``fit.model``, ``fit.likelihood`` and ``fit.priors`` before running.
After changing model limits/fixed flags, call ``fit.configure_priors()`` again.
Use a fresh label when changing a saved run's prior transform.
"""
from pathlib import Path
import json
from time import perf_counter

import bilby
from bilby.core.utils import random
import numpy as np
import torch

from BASIS.models import base
from BASIS.modules.likelihood import ModelLikelihood
from BASIS.modules.native_dynesty import NativeDynamicDynesty
from BASIS.modules.priors import order_interchangeable_priors
from BASIS.modules.utils import to_numpy


class _NumpyModel(base.BaseModel):
    """Keep sampler outputs NumPy without changing models elsewhere in a notebook."""

    def sample_vis(self, uv, ttype='analytical'):
        return to_numpy(super().sample_vis(uv, ttype=ttype))

    def sky_map(self):
        return to_numpy(super().sky_map())


def periodic_angle_boundary(parameter, limits, enabled=True):
    """Wrap recognized angles only when their prior spans one complete period."""
    name = parameter.split('_', 1)[-1]
    period = np.pi if name == 'pa' else 2*np.pi if name == 'phi' or name.endswith('_phase') else None
    if enabled and period is not None and np.isclose(limits[1] - limits[0], period):
        return 'periodic'
    return None


class BilbyHelper:
    """Prepare a fit from an ObsData or UVFITS path and run it with Bilby.

    ``outdir`` defaults to ``bilby_outdir/<label>``. ``likelihood_kwargs`` are
    passed to ModelLikelihood (image size, noise, averaging, covariance, etc.).
    ``sampler_kwargs`` are passed to Bilby; dynamic_dynesty uses the project's
    native checkpoint adapter. ``run_kwargs`` overrides options for one run.
    NumPy output mode also limits Torch to one numerical thread per worker.
    """

    def __init__(self, obs, models, dterms=None, *, label='test', outdir=None,
                 sampler='dynamic_dynesty', npoints=250, ncpu=1, seed=123,
                 param_file=None, likelihood_kwargs=None, sampler_kwargs=None,
                 order_components=True, order_parameter=None, periodic_angles=False,
                 brightest_value=10, closure_intensity_min=0.001,
                 closure_intensity_max=100, allow_torch_outputs=False):
        
        random.seed(seed)

        if not allow_torch_outputs:
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass  # Torch may already have started parallel work in a notebook.

        if isinstance(obs, (str, Path)):
            import ehtim as eh
            obs = eh.obsdata.load_uvfits(str(obs))

        models = [name.strip() for name in models.split(',')] if isinstance(models, str) else list(models)

        options = dict(likelihood_kwargs or {})
        self.likelihood = ModelLikelihood(obs=obs, model_names=None,
                                          dterms={'vis': 100} if dterms is None else dterms,
                                          **options)
        
        model_class = base.BaseModel if allow_torch_outputs else _NumpyModel
        self.model = model_class(model_list=models, dim=self.likelihood.imgdim,
                                 fov=self.likelihood.fov, random_seed=seed)
        
        if param_file is not None:
            self.model.load_params_from_file(param_file)

        self.likelihood.model_names = models
        self.likelihood.model = self.model
        self.likelihood.params = self.model.params
        self.likelihood.configure_closure_model(brightest_value=brightest_value,
                                                minimum=closure_intensity_min,
                                                maximum=closure_intensity_max)
        self.order_components = order_components
        self.order_parameter = order_parameter
        self.periodic_angles = periodic_angles
        
        self.configure_priors()
        self.sampler = sampler
        self.run_options = dict(likelihood=self.likelihood, priors=self.priors,
                                npoints=npoints, outdir=str(outdir or Path('bilby_outdir') / label),
                                label=label, npool=ncpu, check_point_plot=False,
                                **dict(sampler_kwargs or {}))
        self._validate_sampler_options(self.run_options)
        self.result = None

    def configure_priors(self):
        """Rebuild uniform/fixed priors and ordering after model configuration."""
        priors = {}
        for key in self.model.params:
            if self.model.param_fixed[key]:
                prior = bilby.core.prior.DeltaFunction(name=key, peak=self.model.params[key])
            else:
                limits = self.model.param_limits[key]
                prior = bilby.core.prior.Uniform(
                    name=key, minimum=limits[0], maximum=limits[1],
                    boundary=periodic_angle_boundary(key, limits, self.periodic_angles))
            priors[key] = prior
            print(f' - Prior for {key}: {prior}')
        self.ordered_groups = []
        if self.order_components:
            priors, self.ordered_groups = order_interchangeable_priors(
                self.model, priors, self.order_parameter)
            for keys in self.ordered_groups:
                relation = ' > ' if keys[0].endswith('_I0') else ' < '
                print(' - Ordered interchangeable components: ' + relation.join(keys))
        self.priors = priors
        if hasattr(self, 'run_options'):
            self.run_options['priors'] = priors
        return priors

    def _validate_sampler_options(self, options):
        sample, bound = options.get('sample'), options.get('bound')
        proposals = options.get('proposals')
        if any(options.get(key) is not None for key in ('sample', 'bound', 'proposals')):
            if self.sampler not in ('dynesty', 'dynamic_dynesty'):
                raise ValueError('Dynesty proposal/bound options require dynesty or dynamic_dynesty')
        if sample == 'rslice':
            if proposals is not None or bound not in (None, 'multi'):
                raise ValueError('rslice requires bound=multi and no walk proposals')
            options['bound'] = 'multi'
        if proposals and 'diff' in proposals and bound not in (None, 'live', 'live-multi'):
            raise ValueError('diff proposals require bound=live or live-multi')
        if sample == 'acceptance-walk' and options.get('naccept', 60) < 1:
            raise ValueError('naccept must be positive')

    def _check_saved_priors(self, options):
        """Prevent reusing live points generated by a different prior transform."""
        outdir, label = Path(options['outdir']), options['label']
        ordering_file = outdir / 'component_ordering.json'
        spec = dict(groups=self.ordered_groups,
                    priors={key: repr(prior) for key, prior in options['priors'].items()})
        saved = json.loads(ordering_file.read_text()) if ordering_file.exists() else None
        existing = any((outdir / f'{label}{suffix}').exists()
                       for suffix in ('_resume.pickle', '_native_dynesty.save')) or any(
                           outdir.glob(f'{label}_result.*'))
        if existing and saved != spec and (self.ordered_groups or saved):
            raise ValueError('Component ordering/prior transform differs from saved results. Use a fresh label.')
        outdir.mkdir(parents=True, exist_ok=True)
        ordering_file.write_text(json.dumps(spec, indent=2))

    def run(self, **run_kwargs):
        """Run/resume sampling and return Bilby's Result (also saved as result)."""
        options = {**self.run_options, 'priors': self.priors, **run_kwargs}
        self._validate_sampler_options(options)
        self._check_saved_priors(options)
        # Bilby's custom-class dispatcher requires an instance in our environment.
        sampler = NativeDynamicDynesty(**options) if self.sampler == 'dynamic_dynesty' else self.sampler
        self.result = bilby.run_sampler(sampler=sampler, **options)
        self.run_options = options
        return self.result

    def benchmark(self, calls=100):
        """Time prior-drawn likelihood evaluations after warming data caches."""
        if calls < 1:
            raise ValueError('calls must be positive')
        priors = self.priors if isinstance(self.priors, bilby.core.prior.ConditionalPriorDict) else bilby.core.prior.PriorDict(self.priors)
        proposals = [priors.sample() for _ in range(calls)]
        self.likelihood.log_likelihood(proposals[0])
        start = perf_counter()
        for parameters in proposals:
            self.likelihood.log_likelihood(parameters)
        elapsed = perf_counter() - start
        print(f'{calls} likelihood calls in {elapsed:.3f} s ({1000 * elapsed / calls:.3f} ms/call)')
        return dict(calls=calls, seconds=elapsed, ms_per_call=1000 * elapsed / calls)

    def plot_results(self, result=None):
        """Save corner, median-posterior image, and data/model comparison plots."""
        import matplotlib.pyplot as plt
        result = self.result if result is None else result
        if result is None:
            raise ValueError('Run sampling or supply a Bilby result before plotting.')
        parameters = self.model.params.copy()
        parameters.update({key: value for key, value in result.posterior.median().items()
                           if key in parameters})
        result.plot_corner()
        model = base.BaseModel(params=parameters, model_list=self.model.model_list,
                               dim=self.model.dim, fov=self.model.fov)
        prefix = Path(self.run_options['outdir']) / self.run_options['label']
        fig, ax = plt.subplots()
        fov = model.fov
        ax.imshow(to_numpy(model.sky_map()), extent=[fov/2, -fov/2, -fov/2, fov/2],
                  cmap='afmhot', interpolation='gaussian')
        fig.savefig(f'{prefix}_image.png', dpi=200)
        plt.close(fig)
        self.likelihood.plot_all(parameters, save_path=f'{prefix}_data_model_comparison.png')
        return parameters
