import numpy as np
import ehtim as eh
import bilby
from bilby.core.utils import random
import BASIS.modules.likelihood as likelihood
import argparse

import matplotlib.pyplot as plt
from BASIS.models import base

try:
    import torch
except ImportError:
    torch = None


def _to_numpy_output(x):
    """Convert torch outputs to numpy so multiprocessing passes plain arrays."""
    if torch is not None and torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return x


def force_numpy_model_outputs():
    """Patch BaseModel methods to always return numpy objects."""
    original_sample_vis = base.BaseModel.sample_vis
    original_sky_map = base.BaseModel.sky_map

    def sample_vis_numpy(self, uv, ttype='analytical'):
        return _to_numpy_output(original_sample_vis(self, uv, ttype=ttype))

    def sky_map_numpy(self):
        return _to_numpy_output(original_sky_map(self))

    base.BaseModel.sample_vis = sample_vis_numpy
    base.BaseModel.sky_map = sky_map_numpy

def load_obs(uvfits):
    obs = eh.obsdata.load_uvfits(uvfits)
    return obs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bilby model fitting')
    parser.add_argument('--uvfits', type=str, required=True,
                        help='Path to the uvfits file containing the observations')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='Path to the image data (if using image likelihood)')
    parser.add_argument('--models', type=str, required=True,
                        help='Models to fit to the data')
    parser.add_argument('--bilby_label', type=str, default='test',
                        help='Label for the bilby output directory')
    parser.add_argument('--data_terms', type=str, required=True,
                        help='Data terms to include in the likelihood')
    parser.add_argument('--data_weights', type=str, required=True,
                        help='Weights for each data term in the likelihood')
    parser.add_argument('--imgdim', type=int, default=128,
                        help='Image dimension for the model images')
    parser.add_argument('--fov', type=float, default=225,
                        help='Field of view for the model images (in uas)')
    parser.add_argument('--noise_frac', type=float, default=0.0,
                        help='Fractional noise to add to the data term uncertainties')
    parser.add_argument('--static_noise_floor', type=float, default=0.0,
                        help='Noise floor to add to the data term uncertainties')
    parser.add_argument('--noise_factor', type=float, default=1.0,
                        help='Noise factor to scale the data term uncertainties')
    parser.add_argument('--bilby_sampler', type=str, default='dynesty',
                        help='Bilby sampler to use')
    parser.add_argument('--bilby_npoints', type=int, default=250,
                        help='Number of live points for the bilby sampler') 
    parser.add_argument('--random_seed', type=int, default=123,
                        help='Random seed for reproducibility')
    parser.add_argument('--ncpu', type=int, default=1,
                        help='Number of CPU cores to use for bilby sampling')
    parser.add_argument('--allow_torch_outputs', action='store_true', default=False,
                        help='Allow torch tensor outputs from model sampling (not recommended with multiprocessing)')
    parser.add_argument('--param_file', type=str, default=None,
                        help='Path to JSON file containing model parameters and limits (overrides defaults)')

    args = parser.parse_args()
    random.seed(args.random_seed)

    if not args.allow_torch_outputs:
        force_numpy_model_outputs()
        if torch is not None:
            # Keep intra-op threading low when many sampler workers are used.
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        print("Enabled numpy output mode for model sampling.")

    bilby_basedir = "bilby_outdir/"
    outdir = bilby_basedir + args.bilby_label
    bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)

    dterms = {}
    for term, weight in zip(args.data_terms.split(','), args.data_weights.split(',')):
        dterms[term] = float(weight)

    if args.param_file is not None:
        model = base.BaseModel(model_list=args.models.split(','), dim=args.imgdim, fov=args.fov, randomise_params=False, random_seed=args.random_seed)
        model.load_params_from_file(args.param_file)
    else:
        model = base.BaseModel(model_list=args.models.split(','), dim=args.imgdim, fov=args.fov, randomise_params=False, random_seed=args.random_seed)
        
    print()
    print(f'Bilby label: {args.bilby_label}')
    print()
    print(f"Initialising the likelihood with the following parameters:")
    print(f" - Models: {args.models.split(',')}")
    print(f" - Observations: {args.uvfits}")
    print(f" - Image data: {args.image_dir}")
    print(f" - Data terms: {dterms}")
    print(f" - Image dimension: {args.imgdim}")
    print(f" - Field of view: {args.fov} uas")
    print(f" - Static noise floor: {args.static_noise_floor}")
    print(f" - Noise fraction: {args.noise_frac}")
    print(f" - Noise factor: {args.noise_factor}")
    print()

    if args.image_dir is not None:
        img = eh.image.load_fits(args.image_dir)
    else:
        img = None

    custom_likelihood = likelihood.ModelLikelihood(model_names=args.models.split(','), obs=load_obs(args.uvfits), img=img, dterms=dterms,
                                                   imgdim=args.imgdim, fov=args.fov, static_noise_floor=args.static_noise_floor, noise_frac=args.noise_frac, noise_factor=args.noise_factor)

    print()
    priors = dict()
    for param in custom_likelihood.params:
        if model.param_fixed[param]:
            priors[param] = bilby.core.prior.DeltaFunction(
                name=param,
                peak=model.params[param]
            )
            print(f" - Prior for {param}: DeltaFunction({model.params[param]})")
        else:
            priors[param] = bilby.core.prior.Uniform(
                minimum=model.param_limits[param][0],
                maximum=model.param_limits[param][1],
                name=param,
            )
            print(f" - Prior for {param}: Uniform({model.param_limits[param][0]}, {model.param_limits[param][1]})")

    print()
    result = bilby.run_sampler(
        likelihood=custom_likelihood,
        priors=priors,
        sampler=args.bilby_sampler,
        npoints=args.bilby_npoints,
        outdir=outdir,
        label=args.bilby_label,
        npool=args.ncpu,
    )


    print(f"Bilby fitting complete. Results saved to {outdir}/{args.bilby_label}_result.json")
    
    print("Generating corner plot and model image...")
    # result = bilby.result.read_in_result(filename=f'{bilby_basedir}{args.bilby_label}/{args.bilby_label}_result.json')
    model_params = result.posterior.median().to_dict()
    model_params = {k: v for k, v in model_params.items() if k in custom_likelihood.params}
    result.plot_corner() 

    fig, ax = plt.subplots()
    model_instance = base.BaseModel(params=model_params, model_list=args.models.split(','), dim=args.imgdim, fov=args.fov, randomise_params=False, random_seed=args.random_seed)
    ax.imshow(model_instance.sky_map(), extent=[args.fov/2, -args.fov/2, -args.fov/2, args.fov/2], cmap='afmhot', interpolation='gaussian')
    plt.savefig(f"{outdir}/{args.bilby_label}_image.png", dpi=200)
    # plt.close()
    
    print(f"Model image saved to {outdir}/{args.bilby_label}_image.png")

    print()
    custom_likelihood.plot_all(model_params, save_path=f"{outdir}/{args.bilby_label}_data_model_comparison.png")
    print(f"Data/model comparison plot saved to {outdir}/{args.bilby_label}_data_model_comparison.png")

