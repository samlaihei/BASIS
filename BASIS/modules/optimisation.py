import nevergrad as ng
from BASIS.models import base
from BASIS.modules import likelihood
import numpy as np

class ModelOptimisation():
    def __init__(self, model_list=['gauss'], dim=128, fov=225):
        self.model = base.BaseModel(model_list=model_list, dim=dim, fov=fov)
        self.ng_parameterisation = self.build_ng_parameterisation()

    def bhattacharyya_distance(self, p, q):
        """
        Compute the Bhattacharyya distance between two probability distributions.
        """
        # normalise both distributions to unit area
        p = p / np.sum(p)
        q = q / np.sum(q)
        # truncate negative values to zero
        p[p < 0] = 0
        q[q < 0] = 0
        return -np.log(np.sum(np.sqrt(p * q)) + 1e-10)

    def build_ng_parameterisation(self, initial_params=None, model_limits=None, param_fixed=None):
        """
        Build a nevergrad parameterization based on the model limits.
        """
        if initial_params is None:
            initial_params = self.model.params
        if model_limits is None:
            model_limits = self.model.param_limits
        if param_fixed is None:
            param_fixed = self.model.param_fixed

        self.model.params = initial_params
        self.model.param_limits = model_limits
        self.model.param_fixed = param_fixed

        ng_params = []
        for i, (key, limits) in enumerate(model_limits.items()):
            lower, upper = limits
            init_param = initial_params[key] if key in initial_params else (lower + upper) / 2
            if param_fixed:
                if key in param_fixed:
                    if param_fixed[key]:
                        ng_params.append(ng.p.Constant(init_param))
                        continue
            ng_params.append(ng.p.Scalar(init=init_param, lower=lower, upper=upper))
        ng_parameterisation = ng.p.Instrumentation(*ng_params)
        self.ng_parameterisation = ng_parameterisation
        return ng_parameterisation

    def optimise_img(self, objective_img, budget=100, num_workers=1):
        """
        Optimise the model parameters to fit the objective image.
        """
        objective_img = objective_img.regrid_image(self.model.fov*1e-6/206265, self.model.dim, interp='cubic')
        objective_img._imdict['I'] = (objective_img.imarr()/np.sum(objective_img.imarr())).flatten()
        optimiser = ng.optimizers.NGOpt(parametrization=self.ng_parameterisation, budget=budget, num_workers=num_workers)
        optimiser.register_callback("tell", ng.callbacks.ProgressBar())
        recommended_params = optimiser.minimize(lambda *params: self.bhattacharyya_distance(objective_img._imdict['I'], self.model.sky_map(params).flatten()))
        return recommended_params

    def optimise_data(self, likelihood_obj, budget=100, num_workers=1):
        """
        Optimise the model parameters to fit the likelihood object.
        """
        optimiser = ng.optimizers.NGOpt(parametrization=self.ng_parameterisation, budget=budget, num_workers=num_workers)
        optimiser.register_callback("tell", ng.callbacks.ProgressBar())
        recommended_params = optimiser.minimize(lambda *params: -float(likelihood_obj.log_likelihood(dict(zip(self.model.params.keys(), params)))))
        return recommended_params