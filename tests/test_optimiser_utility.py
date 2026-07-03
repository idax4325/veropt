import numpy as np
import pytest
import torch

from veropt.optimiser.optimiser_utility import named_values_to_tensor, format_output_for_objective, \
    get_best_points, get_pareto_optimal_points


def test_get_best_points_simple() -> None:
    variable_values = torch.tensor([
        [0.4, 0.3, 0.7],
        [0.4, 2.4, 0.2],
        [0.1, 1.2, -0.4],
        [3.5, 0.6, 2.1]
    ])
    objective_values = torch.tensor([
        [1.2, 0.5],
        [2.3, 3.4],
        [0.3, 0.5],
        [1.2, 1.4]
    ])

    weights = torch.tensor([0.5, 0.5])

    true_max_index = 1

    best_point = get_best_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights
    )

    assert best_point is not None

    best_variables, best_values, max_index = (
        best_point['variables'], best_point['objectives'], best_point['index']
    )

    assert best_variables is not None, "Something went wrong in this test. Check set-up."
    assert best_values is not None, "Something went wrong in this test. Check set-up."

    assert max_index == true_max_index
    assert torch.equal(best_variables, torch.tensor([0.4, 2.4, 0.2]))
    assert torch.equal(best_values, torch.tensor([2.3, 3.4]))


def test_get_best_points_w_objectives_greater_than() -> None:
    variable_values = torch.tensor([
        [0.4, 0.3, 0.7],
        [0.4, 2.4, 0.2],
        [0.1, 1.2, -0.4],
        [3.5, 0.6, 2.1]
    ])
    objective_values = torch.tensor([
        [1.2, 0.5],
        [0.8, 3.4],
        [0.3, 0.5],
        [1.2, 1.4]
    ])

    weights = torch.tensor([0.5, 0.5])

    true_max_index = 3  # Because we're requiring obj>1

    best_points = get_best_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights,
        objectives_greater_than=1.0
    )

    assert best_points is not None

    best_variables, best_values, max_index = (
        best_points['variables'], best_points['objectives'], best_points['index']
    )

    assert best_variables is not None, "Something went wrong in this test. Check set-up."
    assert best_values is not None, "Something went wrong in this test. Check set-up."

    assert max_index == true_max_index
    assert torch.equal(best_variables, torch.tensor([3.5, 0.6, 2.1]))
    assert torch.equal(best_values, torch.tensor([1.2, 1.4]))


def test_get_pareto_optimal_points() -> None:
    variable_values = torch.tensor([
        [0.4, 0.3, 0.7, -0.3],
        [0.4, 2.4, 0.2, 0.3],
        [0.1, 1.2, -0.4, 0.5],
        [3.5, 0.6, 2.1, -0.4],
        [2.1, -0.3, 0.4, 1.3]
    ])
    objective_values = torch.tensor([
        [1.1, 0.5, 0.3],
        [2.3, 1.2, 0.7],
        [0.3, 0.5, 0.6],
        [1.2, 1.4, 1.1],
        [0.4, 0.6, 2.1]
    ])

    true_pareto_variables = torch.tensor([
        [0.4, 2.4, 0.2, 0.3],
        [3.5, 0.6, 2.1, -0.4],
        [2.1, -0.3, 0.4, 1.3]
    ])

    true_pareto_values = torch.tensor([
        [2.3, 1.2, 0.7],
        [1.2, 1.4, 1.1],
        [0.4, 0.6, 2.1]
    ])

    true_indices = [1, 3, 4]

    pareto_optimal_points = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values
    )

    pareto_variables, pareto_values, pareto_indices = (
        pareto_optimal_points['variables'], pareto_optimal_points['objectives'], pareto_optimal_points['index']
    )

    assert torch.equal(true_pareto_variables, pareto_variables)
    assert torch.equal(true_pareto_values, pareto_values)
    assert np.array_equal(true_indices, pareto_indices)


def test_get_pareto_optimal_points_weights() -> None:
    variable_values = torch.tensor([
        [0.4, 0.3, 0.7, -0.3],
        [0.4, 2.4, 0.2, 0.3],
        [0.1, 1.2, -0.4, 0.5],
        [3.5, 0.6, 2.1, -0.4],
        [2.1, -0.3, 0.4, 1.3]
    ])
    objective_values = torch.tensor([
        [1.1, 0.5, 0.3],
        [2.3, 1.2, 0.7],
        [0.3, 0.5, 0.6],
        [1.2, 1.4, 1.1],
        [0.4, 0.6, 2.1]
    ])

    weights = torch.tensor([1 / 3, 1 / 3, 1 / 3])

    true_pareto_variables = torch.tensor([
        [0.4, 2.4, 0.2, 0.3],
        [3.5, 0.6, 2.1, -0.4],
        [2.1, -0.3, 0.4, 1.3]
    ])

    true_pareto_values = torch.tensor([
        [2.3, 1.2, 0.7],
        [1.2, 1.4, 1.1],
        [0.4, 0.6, 2.1]
    ])

    true_indices = [1, 3, 4]

    pareto_optimal_points = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values,
        weights=weights,
        sort_by_max_weighted_sum=True
    )

    pareto_variables, pareto_values, pareto_indices = (
        pareto_optimal_points['variables'], pareto_optimal_points['objectives'], pareto_optimal_points['index']
    )

    assert torch.equal(true_pareto_variables, pareto_variables)
    assert torch.equal(true_pareto_values, pareto_values)
    assert np.array_equal(true_indices, pareto_indices)


def test_get_pareto_optimal_points_with_noise() -> None:
    # Two objectives, three points:
    #   A = [2.0, 1.0]  -- dominates B in noiseless case (higher on both)
    #   B = [1.8, 0.9]  -- just below A on both objectives
    #   C = [1.0, 2.0]  -- neither dominates the other with A
    #
    # noise_std = [0.2, 0.2], k=1  →  margin = 2 * 1 * 0.2 = 0.4 per objective
    # A certainly dominates B only if A_j > B_j + margin for ALL j:
    #   obj0: 2.0 > 1.8 + 0.4 = 2.2?  No → A does NOT certainly dominate B
    # So B must be kept in the noisy front even though it is noiseless-dominated.
    variable_values = torch.tensor([
        [0.1, 0.2],   # point A
        [0.3, 0.4],   # point B
        [0.5, 0.6],   # point C
    ])
    objective_values = torch.tensor([
        [2.0, 1.0],   # A
        [1.8, 0.9],   # B  (noiseless-dominated by A)
        [1.0, 2.0],   # C
    ])
    noise_std = torch.tensor([0.2, 0.2])

    # Noiseless: B is dominated by A → only A and C on front
    noiseless_result = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values
    )
    noiseless_indices = set(
        [noiseless_result['index']] if isinstance(noiseless_result['index'], int) else noiseless_result['index']
    )
    assert noiseless_indices == {0, 2}, f"Expected noiseless front {{0, 2}}, got {noiseless_indices}"

    # Noisy (k=1, margin=0.4): A does not certainly dominate B → B also kept
    noisy_result = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values,
        noise_std_per_objective=noise_std
    )
    noisy_indices = set(
        [noisy_result['index']] if isinstance(noisy_result['index'], int) else noisy_result['index']
    )
    assert noisy_indices == {0, 1, 2}, f"Expected noisy front {{0, 1, 2}}, got {noisy_indices}"


def test_get_pareto_optimal_points_with_noise_high_epsilon() -> None:
    # Same setup but with epsilon_n_sigma=4: margin = 4 * 0.2 = 0.8
    # Even more conservative → same result (B still not certainly dominated)
    # Also verify that a point that IS certainly dominated gets excluded:
    #   D = [0.5, 0.4]: A certainly dominates D if A_j > D_j + margin for ALL j
    #   obj0: 2.0 > 0.5 + 0.8 = 1.3 ✓   obj1: 1.0 > 0.4 + 0.8 = 1.2?  No → D is kept
    #   But with a point E = [0.0, 0.0]: 2.0 > 0.0 + 0.8 ✓ AND 1.0 > 0.0 + 0.8 ✓ → E is excluded
    variable_values = torch.tensor([
        [0.1, 0.2],   # A
        [0.9, 0.9],   # E  (will be certainly dominated)
    ])
    objective_values = torch.tensor([
        [2.0, 1.0],   # A
        [0.0, 0.0],   # E  (both objectives far below A)
    ])
    noise_std = torch.tensor([0.2, 0.2])

    # With epsilon_n_sigma=4, margin=0.8: A certainly dominates E on both objectives
    result = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values,
        noise_std_per_objective=noise_std,
        epsilon_n_sigma=4.0
    )
    # When only one point is on the front, index is squeezed to a scalar int
    result_indices = {result['index']} if isinstance(result['index'], int) else set(result['index'])
    assert result_indices == {0}, (
        f"Expected only point A (index 0) on the noisy front, got {result_indices}"
    )


def test_format_input_from_objective() -> None:
    expected_amount_points = 4

    variable_names = ['var_1', 'var_2', 'var_3']
    objective_names = ['obj_1', 'obj_2', 'obj_3']

    new_variable_values = {
        'var_3': torch.tensor([0.2, -0.2, -0.1, 2.2]),
        'var_2': torch.tensor([1.2, -1.4, 1.1, 0.2]),
        'var_1': torch.tensor([0.4, 0.3, 0.7, -0.3]),
    }
    new_objective_values = {
        'obj_2': torch.tensor([0.5, -2.1, 0.3, 1.1]),
        'obj_1': torch.tensor([1.1, 0.2, 2.1, 0.4]),
        'obj_3': torch.tensor([0.2, 0.5, 2.1, 2.2]),
    }

    expected_variable_tensor = torch.tensor([
        [0.4, 0.3, 0.7, -0.3],
        [1.2, -1.4, 1.1, 0.2],
        [0.2, -0.2, -0.1, 2.2],
    ])

    expected_objective_values = torch.tensor([
        [1.1, 0.2, 2.1, 0.4],
        [0.5, -2.1, 0.3, 1.1],
        [0.2, 0.5, 2.1, 2.2]
    ])

    expected_variable_tensor = expected_variable_tensor.T
    expected_objective_values = expected_objective_values.T

    (new_variable_values_tensor, new_objective_values_tensor) = named_values_to_tensor(
        new_variable_values=new_variable_values,
        new_objective_values=new_objective_values,
        variable_names=variable_names,
        objective_names=objective_names,
        expected_amount_points=expected_amount_points
    )

    assert torch.equal(expected_variable_tensor, new_variable_values_tensor)
    assert torch.equal(expected_objective_values, new_objective_values_tensor)


def test_format_input_from_objective_too_few_points() -> None:
    expected_amount_points = 4

    variable_names = ['var_1', 'var_2', 'var_3', 'var_4']

    objective_names = ['obj_1', 'obj_2', 'obj_3']

    new_variable_values = {
        'var_3': torch.tensor([0.2, -0.2, -0.1]),
        'var_2': torch.tensor([1.2, -1.4, 1.1]),
        'var_1': torch.tensor([0.4, 0.3, 0.7]),
        'var_4': torch.tensor([-0.5, 0.7, 2.0]),
    }
    new_objective_values = {
        'obj_2': torch.tensor([0.5, -2.1, 0.3]),
        'obj_1': torch.tensor([1.1, 0.2, 2.1]),
        'obj_3': torch.tensor([0.2, 0.5, 2.1]),
    }

    with pytest.raises(AssertionError):
        named_values_to_tensor(
            new_variable_values=new_variable_values,
            new_objective_values=new_objective_values,
            variable_names=variable_names,
            objective_names=objective_names,
            expected_amount_points=expected_amount_points
        )


def test_format_output_for_objective() -> None:
    variable_names = ['var_1', 'var_2', 'var_3']

    suggested_variables_tensor = torch.tensor([
        [0.4, 0.3, 0.7, -0.3],
        [1.2, -1.4, 1.1, 0.2],
        [0.2, -0.2, -0.1, 2.2]
    ])

    suggested_variables_tensor = suggested_variables_tensor.T

    expected_suggested_variables_dict = {
        'var_1': torch.tensor([0.4, 0.3, 0.7, -0.3]),
        'var_2': torch.tensor([1.2, -1.4, 1.1, 0.2]),
        'var_3': torch.tensor([0.2, -0.2, -0.1, 2.2]),
    }

    suggested_variables_dict = format_output_for_objective(
        suggested_variables=suggested_variables_tensor,
        variable_names=variable_names,
    )

    # Converting to list makes it sensitive to the order of the keys
    assert list(suggested_variables_dict.keys()) == list(expected_suggested_variables_dict.keys())

    for name, tensor in expected_suggested_variables_dict.items():
        assert torch.equal(suggested_variables_dict[name], tensor)


def test_get_nadir_point() -> None:
    # TODO: Implement

    pass


def test_format_input_from_objective_with_scalar_floats() -> None:
    expected_amount_points = 1

    variable_names = ['var_1', 'var_2', 'var_3']
    objective_names = ['obj_1', 'obj_2']

    new_variable_values = {
        'var_1': 0.4,
        'var_2': 1.2,
        'var_3': 0.2,
    }
    new_objective_values = {
        'obj_1': 1.1,
        'obj_2': 0.5,
    }

    new_variable_values_tensor, new_objective_values_tensor = named_values_to_tensor(
        new_variable_values=new_variable_values,
        new_objective_values=new_objective_values,
        variable_names=variable_names,
        objective_names=objective_names,
        expected_amount_points=expected_amount_points
    )

    expected_variable_tensor = torch.tensor([[0.4, 1.2, 0.2]])
    expected_objective_tensor = torch.tensor([[1.1, 0.5]])

    assert torch.equal(new_variable_values_tensor, expected_variable_tensor)
    assert torch.equal(new_objective_values_tensor, expected_objective_tensor)
