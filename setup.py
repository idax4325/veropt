from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "torch",
    "gpytorch",
    "botorch",
    "dill",
    "scipy",
    "plotly",
    "kaleido",
    "dash",
    "pydantic",
    "xarray",
    "tqdm"
]

extras_require = {
}

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='veropt',
    version='1.3.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'veropt.optimiser': ['default_settings.json'],
    },
    python_requires='>3.13',
    url='https://github.com/aster-stoustrup/veropt',
    license='OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    author='Aster Stoustrup',
    author_email='aster.stoustrup@gmail.com',
    description='User-friendly Bayesian Optimisation for computationally expensive problems',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require
)
