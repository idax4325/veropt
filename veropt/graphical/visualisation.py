from __future__ import annotations

from itertools import islice
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import plotly.graph_objs as go
import torch
from dash import Dash, Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots

from veropt.graphical._model_visualisation import (
    _fill_model_prediction_from_optimiser, _plot_prediction_grid, _plot_prediction_surface,
    choose_plot_point, _add_labels, _calculate_grid_model_matrix
)
from veropt.graphical._overview import plot_point_overview_separate_subplots, _plot_progression
from veropt.graphical._pareto_front import _plot_pareto_front_grid, _plot_pareto_front
from veropt.graphical._table import (
    _build_table, _plot_table, _save_table_as_csv, _plot_bounds_table
)
from veropt.graphical._visualisation_utility import (
    ModelPredictionContainer, get_point_from_number
)
from veropt.optimiser.acquisition_optimiser import (
    ProximityPunishmentSequentialOptimiser, _calculate_proximity_punished_acquisition_values
)
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_utility import get_best_points, get_pareto_optimal_points, list_with_floats_to_string
from veropt.optimiser.prediction import BotorchPredictor
from veropt.optimiser.utility import DataShape


def plot_point_overview(
        optimiser: BayesianOptimiser,
        points: Literal['all', 'pareto-optimal', 'bayes', 'suggested', 'best'] = 'all',
        normalised: bool = False
) -> go.Figure:

    n_objectives = optimiser.n_objectives

    shown_inds = None

    if normalised:
        variable_values = optimiser.evaluated_variable_values.tensor
        objective_values = optimiser.evaluated_objective_values.tensor
        bounds = optimiser.bounds.tensor
        reference_point = None

    else:
        variable_values = optimiser.evaluated_variables_real_units
        objective_values = optimiser.evaluated_objectives_real_units
        bounds = optimiser.bounds_real_units
        reference_point = optimiser.reference_point

    if points == 'all':
        pass

    elif points == 'bayes':

        shown_inds = np.arange(optimiser.n_initial_points, optimiser.n_points_evaluated).tolist()

    elif points == 'suggested':

        if not normalised:
            raise NotImplementedError()

        assert optimiser.suggested_points, "Must have active suggested points to choose this option"

        suggested_points = optimiser.suggested_points

        variable_values = suggested_points.variable_values

        assert suggested_points.predicted_objective_values is not None, (
            "Must have calculated predictions for the suggested points before calling this function to plot them."
            "(If the model is trained, the optimiser should do this automatically)."
        )

        objective_values = suggested_points.predicted_objective_values['mean']

    elif points == 'best':

        # TODO: Might be optimal to open all points but mark the best ones or make them visible or something

        best_indices = []

        best_points_general = get_best_points(
            variable_values=variable_values,
            objective_values=objective_values,
            weights=optimiser.settings.objective_weights
        )

        assert best_points_general is not None, "Failed to find best points"

        best_indices.append(best_points_general['index'])

        for objective_index in range(n_objectives):

            best_points_for_objective = get_best_points(
                variable_values=variable_values,
                objective_values=objective_values,
                weights=optimiser.settings.objective_weights,
                best_for_objecive_index=objective_index
            )

            assert best_points_for_objective is not None, f"Failed to find best points for objective {objective_index}"

            best_indices.append(
                best_points_for_objective['index']
            )

        shown_inds = np.unique(best_indices).tolist()  # type: ignore[assignment]  # checking below
        assert type(shown_inds) is list
        assert type(shown_inds[0]) is str

    elif points == 'pareto-optimal':

        pareto_points = get_pareto_optimal_points(
            variable_values=variable_values,
            objective_values=objective_values,
            weights=optimiser.settings.objective_weights
        )

        shown_inds = pareto_points['index']

    else:
        raise ValueError()

    objective_names = optimiser.objective.objective_names
    variable_names = optimiser.objective.variable_names

    # Note: Disabled this for now because it just feels a little unnecessary to continue to take care of this
    # when it's probably unlikely to be used. Should maybe just delete eventually?
    #   - Counter-point could be that for large amounts of parameters, *maybe* it's easier to use...?
    #
    # if normalised:
    #     figure = _plot_point_overview(
    #         variable_values=variable_values,
    #         objective_values=objective_values,
    #         objective_names=objective_names,
    #         variable_names=variable_names,
    #         shown_indices=shown_inds
    #     )
    #
    # else:

    figure = plot_point_overview_separate_subplots(
        variable_values=variable_values,
        objective_values=objective_values,
        objective_names=objective_names,
        variable_names=variable_names,
        bounds=bounds,
        shown_indices=shown_inds,
        reference_point=reference_point
    )

    return figure


def plot_progression(
        optimiser: BayesianOptimiser,
        normalised: bool = False
) -> go.Figure:

    if normalised is False:
        objective_values = optimiser.evaluated_objectives_real_units
        reference_point = optimiser.reference_point

    else:
        objective_values = optimiser.evaluated_objective_values.tensor
        reference_point = None

    figure = _plot_progression(
        objective_values=objective_values,
        objective_names=optimiser.objective.objective_names,
        n_initial_points=optimiser.n_initial_points,
        reference_point=reference_point
    )

    return figure


def plot_pareto_front_grid(
        optimiser: BayesianOptimiser,
        normalised: bool = False,
        uncertainty_style: str = 'ellipse'
) -> go.Figure:

    if optimiser.return_normalised_data and normalised is False:
        variable_values = optimiser.evaluated_variables_real_units
        objective_values = optimiser.evaluated_objectives_real_units
        suggested_points = optimiser.suggested_points_real_units
        reference_point = optimiser.reference_point

    else:
        variable_values = optimiser.evaluated_variable_values.tensor
        objective_values = optimiser.evaluated_objective_values.tensor
        suggested_points = optimiser.suggested_points
        reference_point = None

    noise_std_per_objective = optimiser._noise_std_tensor

    pareto_optimal_indices = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values,
        noise_std_per_objective=noise_std_per_objective,
    )['index']

    objective_names = optimiser.objective.objective_names

    figure = _plot_pareto_front_grid(
        objective_values=objective_values,
        objective_names=objective_names,
        pareto_optimal_indices=pareto_optimal_indices,
        n_initial_points=optimiser.n_initial_points,
        suggested_points=suggested_points,
        reference_point=reference_point,
        noise_std_per_objective=noise_std_per_objective,
        uncertainty_style=uncertainty_style,  # type: ignore[arg-type]
        return_figure=True
    )

    return figure


def plot_pareto_front(
        optimiser: BayesianOptimiser,
        plotted_objective_indices: list[int],
        normalised: bool = False,
        uncertainty_style: str = 'ellipse'
) -> go.Figure:

    if optimiser.return_normalised_data and normalised is False:
        variable_values = optimiser.evaluated_variables_real_units
        objective_values = optimiser.evaluated_objectives_real_units
        suggested_points = optimiser.suggested_points_real_units
        reference_point = optimiser.reference_point

    else:
        variable_values = optimiser.evaluated_variable_values.tensor
        objective_values = optimiser.evaluated_objective_values.tensor
        suggested_points = optimiser.suggested_points
        reference_point = None

    noise_std_per_objective = optimiser._noise_std_tensor

    pareto_optimal_indices = get_pareto_optimal_points(
        variable_values=variable_values,
        objective_values=objective_values,
        noise_std_per_objective=noise_std_per_objective,
    )['index']

    figure = _plot_pareto_front(
        objective_values=objective_values,
        pareto_optimal_indices=pareto_optimal_indices,
        plotted_objective_indices=plotted_objective_indices,
        objective_names=optimiser.objective.objective_names,
        n_initial_points=optimiser.n_initial_points,
        suggested_points=suggested_points,
        reference_point=reference_point,
        noise_std_per_objective=noise_std_per_objective,
        uncertainty_style=uncertainty_style,  # type: ignore[arg-type]
        return_figure=True
    )

    return figure


def plot_prediction_grid(
        optimiser: BayesianOptimiser,
        model_prediction_container: Optional[ModelPredictionContainer] = None,
        evaluated_point: Optional[Union[torch.Tensor, int, str]] = None,
        plot_acquisition: bool = False,
        n_calculated_points: Optional[int] = None,
        normalised: bool = False
) -> go.Figure:

    if normalised is False and optimiser.return_normalised_data:
        variable_values = optimiser.evaluated_variables_real_units
        objective_values = optimiser.evaluated_objectives_real_units
        suggested_points = optimiser.suggested_points_real_units
        reference_point = optimiser.reference_point

    else:
        variable_values = optimiser.evaluated_variable_values.tensor
        objective_values = optimiser.evaluated_objective_values.tensor
        suggested_points = optimiser.suggested_points
        reference_point = None

    objective_names = optimiser.objective.objective_names
    variable_names = optimiser.objective.variable_names

    n_variables = variable_values.shape[DataShape.index_dimensions]

    # Only used if calculating new point (make cleaner)
    title_extension = ''

    if model_prediction_container is None:
        model_prediction_container = ModelPredictionContainer(
            normalised=normalised
        )

    else:
        # Could do more checks to make sure this is consistent but this will probably catch most potential errors
        assert model_prediction_container.normalised == normalised

    if isinstance(evaluated_point, str):

        evaluated_point, title_extension = choose_plot_point(
            optimiser=optimiser,
            normalised=normalised,
            point_selection=evaluated_point
        )

    elif isinstance(evaluated_point, int):

        title_extension = f' at point {evaluated_point}'

        evaluated_point = get_point_from_number(
            point_number=evaluated_point,
            variable_values=variable_values,
            suggested_points=suggested_points
        )

    if evaluated_point is None:
        # I guess there's a non-caught case where no point was chosen but the auto-selected point is already calculated
        calculate_new_predictions = True

        evaluated_point, title_extension = choose_plot_point(
            optimiser=optimiser,
            normalised=normalised
        )

    elif evaluated_point in model_prediction_container:
        calculate_new_predictions = False

    elif evaluated_point not in model_prediction_container:
        calculate_new_predictions = True

    else:
        raise RuntimeError("Unexpected error.")

    if calculate_new_predictions:

        for var_ind in range(n_variables):

            calculated_prediction = _fill_model_prediction_from_optimiser(
                optimiser=optimiser,
                variable_index=var_ind,
                evaluated_point=evaluated_point,
                title=title_extension,
                calculate_acquisition=plot_acquisition,
                n_calculated_points=n_calculated_points,
                normalised=normalised
            )

            if optimiser.suggested_points and plot_acquisition:

                if type(optimiser.predictor) is BotorchPredictor:

                    if type(optimiser.predictor.acquisition_optimiser) is ProximityPunishmentSequentialOptimiser:

                        punished_acquisition_values = _calculate_proximity_punished_acquisition_values(
                            proximity_punish_optimiser=optimiser.predictor.acquisition_optimiser,
                            acquisition_function=optimiser.predictor.acquisition_function,
                            normaliser_variables=optimiser.get_normaliser_function_variables(),
                            evaluated_point=calculated_prediction.point,
                            variable_index=var_ind,
                            variable_array=calculated_prediction.variable_array,
                            suggested_points_variables=optimiser.suggested_points.variable_values,
                            normalised=normalised
                        )

                        calculated_prediction.add_modified_acquisition_values(
                            modified_acquisition_values=punished_acquisition_values
                        )

            model_prediction_container.add_data(
                model_prediction=calculated_prediction
            )

    if evaluated_point is None:
        evaluated_point = calculated_prediction.point

    figure = _plot_prediction_grid(
        model_prediction_container=model_prediction_container,
        evaluated_point=evaluated_point,
        variable_values=variable_values,
        objective_values=objective_values,
        objective_names=objective_names,
        variable_names=variable_names,
        suggested_points=suggested_points,
        reference_point=reference_point,
        plot_acquisition=plot_acquisition
    )

    return figure


def run_prediction_grid_app(
        optimiser: BayesianOptimiser,
        normalised: bool = False
) -> None:

    @callback(
        Output('prediction-grid', 'figure'),
        Input('button-go-to-point', 'n_clicks'),
        State('dropdown-points', 'value'),
    )
    def update_x_timeseries(
            n_clicks: int,
            point_index: int
    ) -> go.Figure:

        if point_index is None:
            raise PreventUpdate()

        else:

            chosen_point = variable_values[point_index]

            figure = plot_prediction_grid(
                optimiser=optimiser,
                model_prediction_container=model_prediction_container,
                evaluated_point=chosen_point
            )

            assert figure is not None

        return figure

    n_points_evaluated = optimiser.n_points_evaluated

    if optimiser.suggested_points is None:

        variable_values = optimiser.evaluated_variable_values.tensor
        point_names = [f"Point. {point_no}" for point_no in range(0, n_points_evaluated)]

    else:

        n_suggested_points: int = optimiser.n_evaluations_per_step

        variable_values = torch.concat([
            optimiser.evaluated_variable_values.tensor,
            optimiser.suggested_points.variable_values
        ])

        suggested_point_names = [f"Suggested point no. {point_no}" for point_no in range(n_suggested_points)]

        point_names = (
            [f"Point no. {point_no}" for point_no in range(n_points_evaluated)] + suggested_point_names
        )

    model_prediction_container = ModelPredictionContainer(
        normalised=normalised
    )

    fig_1 = plot_prediction_grid(
        optimiser=optimiser,
        model_prediction_container=model_prediction_container
    )

    dropdown_options = [{'label': point_names[i], 'value': i} for i in range(len(point_names))]

    app = Dash()

    app.layout = html.Div([  # type: ignore[misc]
        html.Div([
            dcc.Graph(
                id='prediction-grid',
                figure=fig_1,
                style={'height': '800px'}
            )
        ]),
        html.Div([
            html.Button(
                'Go to point',
                id='button-go-to-point',
                n_clicks=0
            ),
            dcc.Dropdown(
                id='dropdown-points',
                options=dropdown_options  # type: ignore[arg-type]
            )
        ])
    ])

    app.run()


def plot_prediction_surface(
        optimiser: BayesianOptimiser,
        variable_x: Union[int, str],
        variable_y: Union[int, str],
        objective: Union[int, str],
        evaluated_point: Optional[Union[torch.Tensor, int, str]] = None,
        normalised: bool = False,
        n_points_per_dimension: int = 200,
        figure: Optional[go.Figure] = None,
        row_col: Optional[tuple] = None,
        camera: Optional[dict[Literal['eye', 'center', 'up'], dict[Literal['x', 'y', 'z'], float]]] = None
) -> go.Figure:

    if normalised is False:
        variable_values = optimiser.evaluated_variables_real_units

    else:
        variable_values = optimiser.evaluated_variable_values.tensor

    if evaluated_point is None or isinstance(evaluated_point, str):

        evaluated_point, title = choose_plot_point(
            optimiser=optimiser,
            normalised=normalised,
            point_selection=evaluated_point if isinstance(evaluated_point, str) else None
        )

    elif isinstance(evaluated_point, int):
        evaluated_point = get_point_from_number(
            point_number=evaluated_point,
            variable_values=variable_values,
            suggested_points=None
        )

    if normalised is False:
        bounds = optimiser.bounds_real_units

    else:
        bounds = optimiser.bounds.tensor

    if isinstance(variable_x, str):
        variable_x = optimiser.objective.variable_names.index(variable_x)

    if isinstance(variable_y, str):
        variable_y = optimiser.objective.variable_names.index(variable_y)

    if isinstance(objective, str):
        objective = optimiser.objective.objective_names.index(objective)

    grid_x, grid_y, prediction_objective_matrix = _calculate_grid_model_matrix(
        bounds=bounds,
        variable_x_index=variable_x,
        variable_y_index=variable_y,
        objective_index=objective,
        evaluated_point=evaluated_point,
        predictor=optimiser.predictor,
        normalised=normalised,
        n_points_per_dimension=n_points_per_dimension
    )

    # TODO: This is wrong when we know the evaluated value of the point
    evaluated_point_objective_value = optimiser.predictor.predict_values(
        variable_values=evaluated_point,
        normalised=normalised
    )['mean']

    if normalised is False:
        point_variable_values = optimiser.evaluated_variables_real_units
        point_objective_values = optimiser.evaluated_objectives_real_units[:, objective]
    else:
        point_variable_values = optimiser.evaluated_variable_values.tensor
        point_objective_values = optimiser.evaluated_objective_values[:, objective].tensor

    x_axis_title = optimiser.objective.variable_names[variable_x]
    y_axis_title = optimiser.objective.variable_names[variable_y]
    z_axis_title = optimiser.objective.objective_names[objective]

    if normalised:
        x_axis_title += ' (normalised)'
        y_axis_title += ' (normalised)'
        z_axis_title += ' (normalised)'

    figure = _plot_prediction_surface(
        prediction_objective_matrix=prediction_objective_matrix,
        prediction_grid_x=grid_x,
        prediction_grid_y=grid_y,
        point_variable_values=point_variable_values,
        point_objective_values=point_objective_values,
        evaluated_point=evaluated_point,
        evaluated_point_objective_value=float(evaluated_point_objective_value[0, objective].detach()),
        variable_x_index=variable_x,
        variable_y_index=variable_y,
        x_axis_title=x_axis_title,
        y_axis_title=y_axis_title,
        z_axis_title=z_axis_title,
        figure=figure,
        row_col=row_col
    )

    if camera is not None:
        # Maybe move to underlying function
        figure.update_layout(scene_camera=camera)

    return figure


def plot_prediction_surface_grid(
        optimiser: BayesianOptimiser,
        objective: Union[int, str],
        evaluated_point: Optional[Union[torch.Tensor, int, str]] = None,
        included_variables: Optional[Union[list[int], list[str]]] = None,
        n_points_per_dimension: int = 200,
        camera: Optional[dict[Literal['eye', 'center', 'up'], dict[Literal['x', 'y', 'z'], float]]] = None,
        normalised: bool = False
) -> go.Figure:

    if included_variables is None:
        if optimiser.objective.n_variables**2 > 50:
            raise ValueError(
                "Too many variables to plot for this graph. Please make a selection of variables and pass them through"
                "'included_variables'."
            )

        n_plotted_variables = optimiser.objective.n_variables
        included_variables = list(range(n_plotted_variables))

    if isinstance(included_variables[0], int):
        _included_variables: list[str] = [
            optimiser.objective.variable_names[variable_index]  # type: ignore[index]  # included_variables is list[int]
            for variable_index in included_variables
        ]

    else:
        _included_variables = included_variables  # type: ignore[assignment]  # Checked for other two options
        n_plotted_variables = len(_included_variables)

    if evaluated_point is None or isinstance(evaluated_point, str):

        evaluated_point, title_extension = choose_plot_point(
            optimiser=optimiser,
            normalised=normalised,
            include_suggested_points=False,
            point_selection=evaluated_point if isinstance(evaluated_point, str) else None
        )

    elif isinstance(evaluated_point, int):
        title_extension = f" at point {evaluated_point}"

    else:
        title_extension = ""

    figure = make_subplots(
        rows=n_plotted_variables - 1,
        cols=n_plotted_variables - 1,
        specs=[[{'type': 'surface'} for c in range(n_plotted_variables - 1)] for r in range(n_plotted_variables - 1)],
        horizontal_spacing=0.01,
        vertical_spacing=0.01,
    )

    plotted_combinations = []

    for variable_no_x, variable_x in enumerate(_included_variables[:-1]):
        for variable_no_y, variable_y in islice(enumerate(_included_variables), 1, len(_included_variables)):

            # TODO: Shuffle things around so we can get the figure + row_col out of the user API

            if variable_x == variable_y:
                pass

            elif any((
                    (variable_x, variable_y) in plotted_combinations,
                    (variable_y, variable_x) in plotted_combinations
            )):
                pass

            else:
                figure = plot_prediction_surface(
                    optimiser=optimiser,
                    variable_x=variable_x,
                    variable_y=variable_y,
                    objective=objective,
                    evaluated_point=evaluated_point,
                    normalised=normalised,
                    n_points_per_dimension=n_points_per_dimension,
                    figure=figure,
                    row_col=(variable_no_y, variable_no_x + 1),
                    camera=camera
                )

                plotted_combinations.append(
                    (variable_x, variable_y)
                )

    if isinstance(objective, int):
        title = optimiser.objective.objective_names[objective]

    else:
        title = objective

    figure.update_layout(
        title={'text': title + title_extension}
    )

    labels_x = _included_variables[:-1]
    labels_y = _included_variables[1:]

    labels_x = [
        label + '<br>' + list_with_floats_to_string(optimiser.objective.get_bounds(label)) for label in labels_x
    ]
    labels_y = [
        label + '<br>' + list_with_floats_to_string(optimiser.objective.get_bounds(label)) for label in labels_y
    ]

    _add_labels(
        figure=figure,
        labels_x=labels_x,
        labels_y=labels_y[::-1]
    )

    return figure


def build_table(
        optimiser: BayesianOptimiser,
        chosen_points: list[int]
) -> dict[Literal['variables', 'objectives'], list[dict[str, str | float | None]]]:

    variable_names = optimiser.objective.variable_names
    objective_names = optimiser.objective.objective_names

    # Use real units instead of normalised values
    evaluated_variable_values = optimiser.evaluated_variables_real_units
    evaluated_objective_values = optimiser.evaluated_objectives_real_units

    reference_variable_values = None
    reference_objective_values = None
    if optimiser.reference_point is not None:
        reference_variable_values = optimiser.reference_point.variable_values[0]
        if optimiser.reference_point.objective_values is not None:
            reference_objective_values = optimiser.reference_point.objective_values[0]

    return _build_table(
        variable_names=variable_names,
        objective_names=objective_names,
        chosen_points=chosen_points,
        evaluated_variable_values=evaluated_variable_values,
        evaluated_objective_values=evaluated_objective_values,
        reference_variable_values=reference_variable_values,
        reference_objective_values=reference_objective_values,
    )


def plot_table(
        optimiser: BayesianOptimiser,
        chosen_points: list[int]
) -> go.Figure:

    table_data = build_table(
        optimiser=optimiser,
        chosen_points=chosen_points
    )

    figure = make_subplots(
        rows=2,
        cols=1,
        specs=[[{'type': 'table'}], [{'type': 'table'}]],
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3]
    )

    main_table = _plot_table(
        table_data=table_data,
        bounds=optimiser.bounds_real_units
    )
    figure.add_trace(main_table.data[0], row=1, col=1)

    bounds_table = _plot_bounds_table(
        variable_names=optimiser.objective.variable_names,
        bounds=optimiser.bounds_real_units
    )
    figure.add_trace(bounds_table, row=2, col=1)

    figure.update_layout(height=800)

    return figure


def save_table_to_csv(
        optimiser: BayesianOptimiser,
        chosen_points: list[int],
        file_path: str | Path
) -> None:

    table_data = build_table(
        optimiser=optimiser,
        chosen_points=chosen_points
    )

    _save_table_as_csv(
        table_data=table_data,
        filepath=file_path
    )
