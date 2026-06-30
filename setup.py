from setuptools import setup, find_packages

setup(
    name='BASIS',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'bilby',
        'ehtim',
        'matplotlib',
        'pandas',
        'tqdm',
        'uncertainties',
        'opencv-python',
        'torch'
    ],
    author='Samuel Lai',
)
