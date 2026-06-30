# BASIS

Bayesian Analytical Shape Inference Software (BASIS) is a lightweight Python toolkit for fitting analytic source-structure models to VLBI data with Bayesian inference.

It provides:

- Geometric source models with analytic or direct visibility sampling.
- Bilby-compatible likelihoods for visibility and closure-based data terms.
- Utilities for image-domain ring diagnostics and morphology measurements.
- Optional torch-backed computation.

## Core Components

### Models

Implemented model families include:

- point
- disk
- gauss
- sdisk
- xsring
- xsringauss
- mring
- pixelgrid

Dynamic aliases are supported for mring and pixelgrid:

- mringN for N Fourier-like ring coefficients, for example mring3.
- pixelgridN for an N x N coarse grid, for example pixelgrid8.

### Likelihoods

ModelLikelihood integrates with bilby and supports weighted combinations of:

- image
- vis
- visamp
- bispec
- cphase
- camp
- logcamp
- ci

## Installation

### Requirements

- Python 3.9+
- numpy
- scipy
- bilby
- ehtim
- matplotlib
- pandas
- tqdm
- uncertainties
- opencv-python
- torch

### Install from source

```bash
cd BASIS
pip install -e .
```

Expected installation time: few minutes

## Quick Start

### 1) Build a model image

```python
from BASIS.models.base import BaseModel

model = BaseModel(
    model_list=["xsringauss", "gauss"],
    dim=128,
    fov=225,
    randomise_params=False,
)

image = model.sky_map()
```

### 2) Evaluate a likelihood against uvfits data

```python
import ehtim as eh
from BASIS.modules.likelihood import ModelLikelihood

obs = eh.obsdata.load_uvfits("path/to/data.uvfits")

like = ModelLikelihood(
    model_names=["xsring"],
    obs=obs,
    imgdim=128,
    fov=225,
    dterms={"ci": 100},
    static_noise_floor=1e-4,
    noise_frac=0.05,
    noise_factor=1.0,
)

logl = like.log_likelihood(like.params)
print(logl)
```

### 3) Run Bayesian inference with bilby

A ready-to-run driver script is provided in examples/runBilby.py.

```bash
cd examples
python runBilby.py \
  --bilby_label="test_xsring" \
  --uvfits="../data/uvfits/test_xsring.uvfits" \
  --models="xsring" \
  --imgdim=128 \
  --fov=225 \
  --ncpu=64 \
  --static_noise_floor=0.0001 \
  --noise_frac=0.05 \
  --noise_factor=1.0 \
  --bilby_sampler="dynesty" \
  --bilby_npoints=250 \
  --random_seed=42 \
  --data_terms="ci" \
  --data_weights="100"
```

Outputs are written under examples/bilby_outdir/ including posterior files, corner plot, model image, and data-model comparison plots.

## Examples

Notebook and script examples are available in examples/:

- Example_BASIS.ipynb
- Example_bilby.ipynb
- Example_ringFitting.ipynb
- runBilby.py
- runBilby.sh

## License

MIT License. See LICENSE.

## Citation

If you use BASIS in research, cite the associated software release and related scientific publications from this project.
