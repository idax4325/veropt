"""Example: Noisy Pareto Front Visualisation.

Shows how to set a known constant observation noise on a multi-objective
problem.  Shaded ellipses on the Pareto front represent ±1σ noise per
objective.  The highlighted front uses noise-aware dominance so borderline
points near the noise floor are not prematurely excluded.

Run
---
    python examples/example_noisy_pareto_front.py
"""

from veropt.optimiser.constructors import bayesian_optimiser
from veropt.optimiser.practice_objectives import DTLZ1
from veropt.graphical.visualisation import plot_pareto_front

objective = DTLZ1(
    n_variables=6,
    n_objectives=2,
    noise_std={'DTLZ1 1': 10, 'DTLZ1 2': 10}
)

optimiser = bayesian_optimiser(
    n_initial_points=80,
    n_bayesian_points=16,
    n_evaluations_per_step=4,
    objective=objective,
    model={'training_settings': {'max_iter': 10}},
    acquisition_optimiser={'optimiser': 'dual_annealing', 'optimiser_settings': {'max_iter': 100}},
)

while optimiser.n_points_evaluated < optimiser.n_initial_points:
    optimiser.run_optimisation_step()

for step_number in range(2):
    print(f"Bayesian step {step_number + 1} / 2")
    optimiser.run_optimisation_step()

# Ellipse style (default)
plot_pareto_front(optimiser=optimiser, plotted_objective_indices=[0, 1]).show()

# Error bars style
plot_pareto_front(
    optimiser=optimiser,
    plotted_objective_indices=[0, 1],
    uncertainty_style='error_bars',
).show()
