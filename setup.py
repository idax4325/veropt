from setuptools import setup, find_packages

install_requires = ["botorch", "dill", "click", "scikit-learn", "scipy", "matplotlib", "numpy"]
extras_require = {
    "gui": ["PySide2"],
    "multi_processing_smp": ["pathos"],
    "mpi": ["mpi4py"]
}

setup(
    name='veropt',
    version='0.4',
    packages=find_packages(),
    url='https://github.com/idax4325/veropt',
    license='OSI Approved :: MIT License',
    author='Ida Stoustrup',
    author_email='Ida.Stoustrup@gmail.com',
    description='Bayesian Optimisation for the Versatile Ocean Simulator (VEROS)',
    install_requires=install_requires,
    extras_require=extras_require
)
