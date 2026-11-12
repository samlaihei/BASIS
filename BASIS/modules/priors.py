"""Normalized prior transforms for interchangeable additive model components."""
import numpy as np
import bilby
from bilby.core.utils import random


class OrderedUniform(bilby.core.prior.Prior):
    """Next order statistic of identical independent uniform coordinates.

    With k values remaining, the next coordinate is lower + (upper-lower)*Beta(1,k),
    where lower is the previous coordinate. The full joint density is n!/width**n
    on the ordered region. Unlike sequential conditional Uniforms, this preserves
    the original prior over unlabeled models and its evidence normalization.
    """

    def __init__(self, minimum, maximum, remaining, previous=None, name=None,
                 latex_label=None, unit=None):
        if maximum <= minimum or not isinstance(remaining, int) or remaining < 1:
            raise ValueError('Require maximum > minimum and positive integer remaining')
        super().__init__(minimum=minimum, maximum=maximum, name=name,
                         latex_label=latex_label, unit=unit)
        self.remaining = remaining
        self.previous = previous
        self.required_variables = [] if previous is None else [previous]
        # ConditionalPriorDict identifies conditional distributions by this attribute.
        self.condition_func = self._condition

    def _condition(self, reference_params, **variables):
        return dict(minimum=self._lower(variables))

    def _lower(self, variables):
        if not variables:
            return self.minimum
        if set(variables) != set(self.required_variables):
            raise ValueError(f'Expected dependencies {self.required_variables}')
        return np.asarray(variables[self.previous])

    def rescale(self, val, **variables):
        lower = self._lower(variables)
        with np.errstate(divide='ignore'):
            fraction = -np.expm1(np.log1p(-np.asarray(val))/self.remaining)
        return lower + (self.maximum-lower)*fraction

    def sample(self, size=None, **variables):
        self.least_recently_sampled = self.rescale(random.rng.uniform(size=size), **variables)
        return self.least_recently_sampled

    def ln_prob(self, val, **variables):
        lower = self._lower(variables)
        val = np.asarray(val)
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.log(self.remaining)-np.log(self.maximum-lower)
            if self.remaining > 1:
                result = result+(self.remaining-1)*np.log((self.maximum-val)/(self.maximum-lower))
        return np.where((val >= lower) & (val <= self.maximum) & (lower < self.maximum), result, -np.inf)

    def prob(self, val, **variables):
        return np.exp(self.ln_prob(val, **variables))

    def cdf(self, val, **variables):
        lower = self._lower(variables)
        with np.errstate(divide='ignore', invalid='ignore'):
            fraction = np.clip((np.asarray(val)-lower)/(self.maximum-lower), 0, 1)
        return 1-(1-fraction)**self.remaining

    def get_instantiation_dict(self):
        return dict(minimum=self.minimum, maximum=self.maximum, remaining=self.remaining,
                    previous=self.previous, name=self.name, latex_label=self.latex_label, unit=self.unit)


def order_interchangeable_priors(model, priors, parameter=None):
    """Return ConditionalPriorDict and ordered key groups, leaving inputs unchanged.

    Only identical Uniform/DeltaFunction component priors are supported; differing
    or custom priors are conservatively left alone. Flux (I0) is ordered
    descending, so the lowest-index component is brightest. If flux is fixed,
    fall back to an ascending free size/position coordinate. Explicit non-flux
    coordinates are also ascending. Sampling never rejects unordered draws.
    """
    result = bilby.core.prior.ConditionalPriorDict(dict(priors))
    ordered_groups = []
    for indices in model.interchangeable_components():
        suffixes = sorted(key.split('_', 1)[1] for key in model.params
                          if key.startswith(f'{indices[0]}_'))
        signatures = []
        for index in indices:
            signature = []
            for suffix in suffixes:
                prior = priors[f'{index}_{suffix}']
                if type(prior) not in (bilby.core.prior.Uniform, bilby.core.prior.DeltaFunction):
                    break
                fields = prior.get_instantiation_dict()
                signature.append((type(prior), {k: v for k, v in fields.items()
                                                if k not in ('name', 'latex_label')}))
            else:
                signatures.append(signature)
        if len(signatures) != len(indices) or any(sig != signatures[0] for sig in signatures[1:]):
            continue
        candidates = [parameter] if parameter else ['I0', 'gauss_sigma', 'Rp', 'radius', 'xoff', 'yoff']
        coordinate = next((name for name in candidates if name in suffixes
                           and type(priors[f'{indices[0]}_{name}']) is bilby.core.prior.Uniform
                           and priors[f'{indices[0]}_{name}'].boundary is None), None)
        if coordinate is None:
            if parameter:
                raise ValueError(f'Cannot order components {indices} by {parameter}: require a free non-periodic Uniform')
            continue
        keys = [f'{index}_{coordinate}' for index in indices]
        # Build ascending order statistics from the last label to the first for
        # flux. This gives descending physical labels without changing the density.
        sampling_keys = keys[::-1] if coordinate == 'I0' else keys
        for position, key in enumerate(sampling_keys):
            prior = priors[key]
            result[key] = OrderedUniform(prior.minimum, prior.maximum, len(keys)-position,
                                         previous=sampling_keys[position-1] if position else None,
                                         name=key, latex_label=prior.latex_label, unit=prior.unit)
        ordered_groups.append(keys)
    return result, ordered_groups
