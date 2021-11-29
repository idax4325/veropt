from setuptools import setup, find_packages

install_requires = ["botorch", "dill", "click", "scikit-learn==0.24.1", "scipy", "matplotlib", "numpy", "xarray"]
extras_require = {
    "gui": ["PySide6"],
    "multi_processing_smp": ["pathos"],
    "mpi": ["mpi4py"]
}

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='veropt',
    version='0.5.1',
    packages=find_packages(),
    url='https://github.com/idax4325/veropt',
    license='OSI Approved :: MIT License',
    author='Ida Stoustrup',
    author_email='Ida.Stoustrup@gmail.com',
    description='Bayesian Optimisation for the Versatile Ocean Simulator (VEROS)',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=install_requires,
    extras_require=extras_require
)
