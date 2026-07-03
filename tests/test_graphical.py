import csv
import tempfile
from pathlib import Path

import torch

from veropt.optimiser.constructors import bayesian_optimiser
from veropt.optimiser.practice_objectives import VehicleSafety
from veropt.graphical.visualisation import save_table_to_csv
from veropt.graphical._pareto_front import _add_pareto_traces_2d


def test_save_table_to_csv() -> None:

    objective = VehicleSafety()

    optimiser = bayesian_optimiser(
        n_initial_points=32,
        n_bayesian_points=16,
        n_evaluations_per_step=4,
        objective=objective,
        verbose=False,
    )

    for _ in range(4):
        optimiser.run_optimisation_step()

    chosen_points = [0, 1, 10, 12]

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "table.csv"

        save_table_to_csv(
            optimiser=optimiser,
            chosen_points=chosen_points,
            file_path=csv_path
        )

        with open(csv_path, newline='') as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

    variable_names = optimiser.objective.variable_names
    objective_names = optimiser.objective.objective_names
    evaluated_variables = optimiser.evaluated_variables_real_units
    evaluated_objectives = optimiser.evaluated_objectives_real_units

    # Split rows into variable rows, separator, and objective rows
    variable_rows = rows[:len(variable_names)]
    separator_row = rows[len(variable_names)]
    objective_rows = rows[len(variable_names) + 1:]

    assert separator_row["Name"] == "Objectives"

    # Check variable rows
    for var_idx, var_name in enumerate(variable_names):
        assert variable_rows[var_idx]["Name"] == var_name
        for point_idx in chosen_points:
            expected_value = float(evaluated_variables[point_idx, var_idx])
            csv_value = float(variable_rows[var_idx][f"point_{point_idx}"])
            assert abs(csv_value - expected_value) < 1e-4, (
                f"Variable '{var_name}' at point {point_idx}: "
                f"expected {expected_value}, got {csv_value}"
            )

    # Check objective rows
    for obj_idx, obj_name in enumerate(objective_names):
        assert objective_rows[obj_idx]["Name"] == obj_name
        for point_idx in chosen_points:
            expected_value = float(evaluated_objectives[point_idx, obj_idx])
            csv_value = float(objective_rows[obj_idx][f"point_{point_idx}"])
            assert abs(csv_value - expected_value) < 1e-4, (
                f"Objective '{obj_name}' at point {point_idx}: "
                f"expected {expected_value}, got {csv_value}"
            )


class TestUncertainParetoFront:

    def test_add_pareto_traces_2d_with_ellipse_noise_adds_noise_trace(self) -> None:
        """With noise and ellipse style, at least one noise trace is added."""
        import plotly.graph_objects as go
        objective_values = torch.rand(20, 3)
        noise_std = torch.tensor([0.1, 0.1, 0.05])
        figure = go.Figure()
        figure = _add_pareto_traces_2d(
            figure=figure,
            objective_values=objective_values,
            objective_index_x=0,
            objective_index_y=1,
            objective_names=['A', 'B', 'C'],
            pareto_optimal_indices=[0, 1, 2],
            n_initial_points=10,
            noise_std_per_objective=noise_std,
            uncertainty_style='ellipse',
        )
        trace_names = [t.name for t in figure.data]
        # At least one noise trace should be present (name starts with 'Noise')
        assert any('Noise' in (name or '') for name in trace_names)

    def test_add_pareto_traces_2d_without_noise_has_no_noise_trace(self) -> None:
        """Without noise, no 'Noise' trace is added."""
        import plotly.graph_objects as go
        objective_values = torch.rand(20, 3)
        figure = go.Figure()
        figure = _add_pareto_traces_2d(
            figure=figure,
            objective_values=objective_values,
            objective_index_x=0,
            objective_index_y=1,
            objective_names=['A', 'B', 'C'],
            pareto_optimal_indices=[0, 1, 2],
            n_initial_points=10,
            noise_std_per_objective=None,
        )
        trace_names = [t.name for t in figure.data]
        assert not any('Noise' in (name or '') for name in trace_names)

    def test_add_pareto_traces_2d_with_error_bars_sets_error_y(self) -> None:
        """With error_bars style, scatter traces should have error_y set."""
        import plotly.graph_objects as go
        objective_values = torch.rand(20, 3)
        noise_std = torch.tensor([0.1, 0.1, 0.05])
        figure = go.Figure()
        figure = _add_pareto_traces_2d(
            figure=figure,
            objective_values=objective_values,
            objective_index_x=0,
            objective_index_y=1,
            objective_names=['A', 'B', 'C'],
            pareto_optimal_indices=[0, 1, 2],
            n_initial_points=10,
            noise_std_per_objective=noise_std,
            uncertainty_style='error_bars',
        )
        scatter_traces = [t for t in figure.data if hasattr(t, 'error_y')]
        assert any(t.error_y is not None for t in scatter_traces)
        # Error bars style should not add ellipse traces
        trace_names = [t.name for t in figure.data]
        assert not any('Noise' in (name or '') for name in trace_names)

    def test_noisy_pareto_dominance_keeps_more_points_than_noiseless(self) -> None:
        """With noise, uncertain points near the boundary should not be discarded.

        We construct a case where point B strictly dominates point A in noiseless
        terms, but the margin is small enough that with noise the dominance is not
        *certain*.  The noisy criterion should keep both points; the noiseless
        criterion should keep only B.
        """
        from veropt.optimiser.optimiser_utility import get_pareto_optimal_points

        # Two points, two objectives (maximisation).
        # B is just slightly better than A on both axes — strictly dominated noiseless.
        a = torch.tensor([[1.0, 1.0]])
        b = torch.tensor([[1.05, 1.05]])
        objective_values = torch.cat([a, b], dim=0)
        variable_values = torch.zeros(2, 1)

        noiseless_result = get_pareto_optimal_points(
            variable_values=variable_values,
            objective_values=objective_values,
        )
        # Noiseless: only B should survive (A is dominated by B).
        # Note: when only 1 point survives, tolist() returns a bare int, not a list.
        noiseless_indices = noiseless_result['index']
        noiseless_indices_list = [noiseless_indices] if isinstance(noiseless_indices, int) else noiseless_indices
        assert len(noiseless_indices_list) == 1
        assert 1 in noiseless_indices_list

        # With noise_std=0.1, the 0.05 gap is well within 2σ=0.2 — A should survive too
        noise_std = torch.tensor([0.1, 0.1])
        noisy_result = get_pareto_optimal_points(
            variable_values=variable_values,
            objective_values=objective_values,
            noise_std_per_objective=noise_std,
        )
        assert len(noisy_result['index']) == 2
