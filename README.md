
# veropt

## Bayesian Optimisation for the Versatile Ocean Simulator

veropt is a Python package that aims to make Bayesian Optimisation easy to approach, inspect and adjust. It was developed for the Versatile Ocean Simulator ([VEROS](https://veros.readthedocs.io/en/latest/)) with the aim of providing a user-friendly optimisation tool to tune ocean simulations to real world data. 

veropt can be used with any optimisation problem but has been developed for expensive optimisation problems with a small amount of evaluations (~100) and the default set-up will probably be most relevant in such a context.

For more information about the package and the methods implemented in it, take a look at my [thesis report](https://nbi.ku.dk/english/theses/masters-theses/ida_lei_stoustrup/Ida_Stoustrup_MSc_Thesis.pdf). 

## Installation

To install veropt with the default dependencies *and* the package utilised by the GUI (PySide2), do the following:

```bash
pip install veropt[gui]
```

Or, in an zsh terminal,

```bash
pip install "veropt[gui]"
```

If you're installing veropt on a cluster and don't need the GUI you can simply do,

```bash
pip install veropt
```

##

Please note that veropt depends on PyTorch. When installing a larger library like that, I would usually recommend using anaconda over pip. To install PyTorch with anaconda, you can run,
```bash
conda install pytorch torchvision -c pytorch
```

You may also want to consider creating a new conda environment before running the PyTorch installation.

##

If you need to run a veropt *experiment* (only relevant when benchmarking an optimisation set-up against random search or comparing different set-ups) and you want to run it in parallel, you will need either pathos or mpi4py. The first-mentioned will be included by doing,

```bash
pip install veropt[multi_processing_smp]
```
This is recommended if you're running experiments on a laptop. 

If you're running experiments on a cluster, you will need mpi instead. Please note that mpi4py installations can be quite complex and it is probably advisable to do a manual installation before installing veropt. But if you're feeling adventurous and want to see if pip can do it, you can run,

```bash
pip install veropt[mpi]
```


## Usage

Below is a simple example of running an optimisation problem with veropt. 

```python
from veropt import BayesOptimiser
from veropt.obj_funcs.test_functions import *
from veropt.gui import veropt_gui

n_init_points = 24
n_bayes_points = 64
n_evals_per_step = 4

obj_func = PredefinedTestFunction("Hartmann")


optimiser = BayesOptimiser(n_init_points, n_bayes_points, obj_func, n_evals_per_step=n_evals_per_step)

veropt_gui.run(optimiser)
```

This example utilises one of the predefined test objective functions found in veropt.obj_funcs.test_functions. 

To use veropt with your own optimisation problem, you will need to create a class that uses the "ObjFunction" class from veropt/optimiser.py as a superclass. Your class must either include a method of running your objective function (YourClass.function()) or a method for both saving parameter values and loading new objective function values (YourClass.saver() and YourClass.loader()).

If you're using veropt with a veros simulation, take a look at veropt/obj_funcs/ocean_sims and the veros simulation examples under examples/ocean_examples.

## The GUI and the Visualisation Tools

<img width="850" alt="GUI" src="https://user-images.githubusercontent.com/33256573/134529054-cfd9a3bb-8641-4cd2-8a11-fc6d7f794e1c.png">

After running the command,


```python
veropt_gui.run(optimiser)
```

You should see a window like the one above. From here, you can show the progress of the optimisation, visualise the predictions of the GP model, change essential parameters of the model or acquisition function and much more. 

##

If you press "Plot predictions" in the GUI, you will encounter a plot like the one below. 

<img width="700" alt="pred1" src="https://github.com/idax4325/veropt/files/7218616/BranninCurrinPrediction_wsust.pdf">

It shows a slice of the function domain, along the axis of a chosen optimisation parameter. You will be able to inspect the model, the acquisition function, as well as the suggested points for the next round of objective function evaluations. If any of this isn't as desired, you simply close the figure and go back to the GUI to modify the optimisation by changing the relevant parameters.

## License

This project uses the [MIT](https://choosealicense.com/licenses/mit/) license.
