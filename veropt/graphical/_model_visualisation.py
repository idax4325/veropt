from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from plotly import graph_objs as go
from plotly.express import colors
from plotly.subplots import make_subplots

from veropt.graphical._visualisation_utility import (
    ModelPrediction, ModelPredictionContainer,
    opacity_for_multidimensional_points, get_continuous_colour
)
from veropt.optimiser.optimiser import BayesianOptimiser
from veropt.optimiser.optimiser_utility import ReferencePoint, SuggestedPoints
from veropt.optimiser.prediction import Predictor
from veropt.optimiser.utility import DataShape


def choose_plot_point(
        optimiser: BayesianOptimiser,
        normalised: bool,
        include_suggested_points: bool = True,
        point_selection: Optional[str] = None
) -> tuple[torch.Tensor, str]:
    """
    Resolve a plot evaluation point from a string selector or default to the best evaluated point.

    point_selection options:
      None / "best"              -> best evaluated point (weighted sum of objectives)
      "best {objective_name}"    -> best point for a specific objective
      "suggested {n}"            -> nth suggested point (1-indexed)
    """

    if point_selection is not None and point_selection.startswith("suggested"):

        assert optimiser.suggested_points is not None, (
            "No active suggested points - cannot use 'suggested N' point selection."
        )
        assert include_suggested_points, (
            "include_suggested_points=False but point_selection references a suggested point"
        )

        parts = point_selection.split()
        assert len(parts) == 2 and parts[1].isdigit(), (
            f"Expected 'suggested N' (1-indexed), got: '{point_selection}'"
        )
        suggested_point_number = int(parts[1])

        if normalised is False:
            assert optimiser.suggested_points_real_units is not None, "Internal error"
            suggested_variable_values = optimiser.suggested_points_real_units.variable_values
        else:
            suggested_variable_values = optimiser.suggested_points.variable_values

        n_suggested = suggested_variable_values.shape[0]
        assert 1 <= suggested_point_number <= n_suggested, (
            f"Suggested point number {suggested_point_number} out of range (1 to {n_suggested})"
        )

        suggested_point_ind = suggested_point_number - 1
        eval_point = deepcopy(suggested_variable_values[suggested_point_ind:suggested_point_ind + 1])
        point_description = f" at suggested point {suggested_point_number}"

    elif point_selection is not None and point_selection.startswith("best "):

        objective_name = point_selection[len("best "):]
        objective_names = optimiser.objective.objective_names
        assert objective_name in objective_names, (
            f"Objective '{objective_name}' not found. Available: {objective_names}"
        )
        objective_index = objective_names.index(objective_name)

        if normalised is False:
            variable_values = optimiser.evaluated_variables_real_units
            objective_values = optimiser.evaluated_objectives_real_units
        else:
            variable_values = optimiser.evaluated_variable_values.tensor
            objective_values = optimiser.evaluated_objective_values.tensor

        best_index = int(objective_values[:, objective_index].argmax())
        eval_point = deepcopy(variable_values[best_index:best_index + 1])
        point_description = f" at the best point for '{objective_name}' (point no. {best_index})"

    else:

        if normalised is False:
            variable_values = optimiser.evaluated_variables_real_units
        else:
            variable_values = optimiser.evaluated_variable_values.tensor

        max_ind = optimiser.get_best_points()['index']
        eval_point = deepcopy(variable_values[max_ind:max_ind + 1])
        point_description = f" at the best known point (point no. {max_ind})"

    return eval_point, point_description


def _fill_model_prediction_from_optimiser(
        optimiser: BayesianOptimiser,
        variable_index: int,
        evaluated_point: torch.Tensor,
        title: str,
        normalised: bool,
        n_calculated_points: Optional[int] = None,
        calculate_acquisition: bool = False,
) -> ModelPrediction:

    if n_calculated_points is None:
        if calculate_acquisition is False:
            n_calculated_points = 1_000
        else:
            n_calculated_points = 200

    if normalised is False:
        bounds = optimiser.bounds_real_units
    else:
        bounds = optimiser.bounds.tensor

    variable_array = torch.linspace(
        start=bounds[0, variable_index],
        end=bounds[1, variable_index],
        steps=n_calculated_points
    )

    all_variables_array = evaluated_point.repeat(len(variable_array), 1)
    all_variables_array[:, variable_index] = variable_array

    if calculate_acquisition:
        acquisition_values = optimiser.predictor.get_acquisition_values(
            variable_values=all_variables_array,
            normalised=normalised
        )
    else:
        acquisition_values = None

    samples = optimiser.predictor.get_samples_from_model(
        variable_values=all_variables_array,
        n_samples=5,
        normalised=normalised
    )

    predicted_objective_values = optimiser.predictor.predict_values(
        variable_values=all_variables_array,
        normalised=normalised
    )

    return ModelPrediction(
        variable_index=variable_index,
        point=evaluated_point,
        title=title,
        variable_array=variable_array,
        predicted_objective_values=predicted_objective_values,
        acquisition_values=acquisition_values,
        samples=samples
    )


def _add_model_traces(
        figure: go.Figure,
        model_prediction: ModelPrediction,
        row_no: int,
        col_no: int,
        objective_index: int,
        legend_group: str
) -> None:

    predicted_values_mean = model_prediction.predicted_values_mean[:, objective_index]
    predicted_values_lower = model_prediction.predicted_values_lower[:, objective_index]
    predicted_values_upper = model_prediction.predicted_values_upper[:, objective_index]

    variance_fill_colour = 'rgba(0.7, 0.7, 0.7, 0.3)'

    figure.add_trace(
        go.Scatter(
            x=model_prediction.variable_array,
            y=predicted_values_upper.detach().numpy(),
            line={'width': 0.0, 'color': variance_fill_colour},
            name='Upper bound prediction',
            legendgroup=legend_group,
            showlegend=False
        ),
        row=row_no, col=col_no
    )

    figure.add_trace(
        go.Scatter(
            x=model_prediction.variable_array,
            y=predicted_values_lower.detach().numpy(),
            fill='tonexty',  # This fills between this and the line above
            line={'width': 0.0, 'color': variance_fill_colour},
            name='Lower bound prediction',
            legendgroup=legend_group,
            showlegend=False,
        ),
        row=row_no, col=col_no
    )

    figure.add_trace(
        go.Scatter(
            x=model_prediction.variable_array,
            y=predicted_values_mean.detach().numpy(),
            line={'color': 'black'},
            name='Mean prediction',
            legendgroup=legend_group,
            showlegend=True if (row_no == 1 and col_no == 1) else False
        ),
        row=row_no, col=col_no
    )

    for sample_no in range(len(model_prediction.samples)):

        show_legend_sample = True if (row_no == 1 and col_no == 1) else False
        show_legend_sample = show_legend_sample and (sample_no == 0)

        figure.add_trace(
            go.Scatter(
                x=model_prediction.variable_array,
                y=model_prediction.samples[sample_no][:, objective_index].detach().numpy(),
                line={'color': "rgba(0.5, 0.5, 0.5, 0.3)"},
                name='Model samples',
                legendgroup='Model samples',
                showlegend=show_legend_sample,
                visible='legendonly'
            ),
            row=row_no, col=col_no
        )


def _plot_prediction_grid(
        model_prediction_container: ModelPredictionContainer,
        evaluated_point: torch.Tensor,
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        objective_names: list[str],
        variable_names: list[str],
        suggested_points: Optional[SuggestedPoints] = None,
        reference_point: Optional[ReferencePoint] = None,
        plot_acquisition: bool = False
) -> go.Figure:

    # TODO: Add option to plot subset of all these
    #   - Could be from var/obj start_ind to var/obj end_ind
    #   - Could be lists of vars and objs
    #   - Could be single var or single obj
    #   - Could be mix of these

    n_evaluated_points = variable_values.shape[DataShape.index_points]

    if suggested_points:
        n_suggested_points = len(suggested_points)

    n_variables = variable_values.shape[1]
    n_objectives = len(objective_names)

    colour_scale = colors.get_colorscale('matter')
    colour_scale_suggested_points = colors.get_colorscale('Emrld')
    # colour_list = colors.sample_colorscale(
    #     colorscale=colour_scale,
    #     samplepoints=n_evaluated_points,
    #     low=0.0,
    #     high=1.0,
    #     colortype='rgb'
    # )

    figure = make_subplots(
        rows=n_objectives,
        cols=n_variables
    )

    for variable_index in range(n_variables):

        model_prediction = model_prediction_container(
            variable_index=variable_index,
            point=evaluated_point
        )

        if suggested_points:
            joint_points = torch.concat([
                variable_values,
                suggested_points.variable_values
            ])
        else:
            joint_points = variable_values

        joint_opacity_list, joint_distance_list = opacity_for_multidimensional_points(
            variable_indices=[variable_index],
            variable_values=joint_points,
            evaluated_point=evaluated_point,
            alpha_min=0.4,
            alpha_max=1.0
        )

        distance_list = joint_distance_list[:n_evaluated_points]
        suggested_point_distance_list = joint_distance_list[n_evaluated_points:]

        marker_type_list = ['circle'] * n_evaluated_points
        marker_size_list = [8] * n_evaluated_points

        evaluated_point_ind = np.where(joint_distance_list == 0.0)[0][0]

        if evaluated_point_ind < n_evaluated_points:
            marker_type_list[evaluated_point_ind] = 'x'
            marker_size_list[evaluated_point_ind] = 14

        if evaluated_point_ind >= n_evaluated_points:
            evaluated_suggested_point_ind = evaluated_point_ind - n_evaluated_points

        else:
            evaluated_suggested_point_ind = None

        colour_list = [get_continuous_colour(colour_scale, float(1 - distance)) for distance in distance_list]

        colour_list_w_opacity = [
            "rgba(" + colour_list[point_no][4:-1] + f", {joint_opacity_list[point_no]})"
            for point_no in range(n_evaluated_points)
        ]

        if suggested_points:

            colour_list_suggested_points = [
                get_continuous_colour(colour_scale_suggested_points, float(1 - distance))
                for distance in suggested_point_distance_list
            ]

            colour_list_suggested_points_w_opacity = [
                (
                    f"rgba("
                    f"{colour_list_suggested_points[point_no][4:-1]}, "
                    f"{joint_opacity_list[n_evaluated_points + point_no]}"
                    f")"
                )
                for point_no in range(n_suggested_points)
            ]

        for objective_index in range(n_objectives):

            # Placing these backwards to make the "y axes" of subplots go positive upwards
            row_no = n_objectives - objective_index
            col_no = variable_index + 1

            # Quick scaling as long as we're just jamming it into this plot
            # acq_func_scaling = np.abs(model_pred_data.acq_fun_vals).max() * 0.5
            acq_func_scaling = 1.0

            _add_model_traces(
                figure=figure,
                model_prediction=model_prediction,
                row_no=row_no,
                col_no=col_no,
                objective_index=objective_index,
                legend_group='model'
            )

            if plot_acquisition:

                assert model_prediction.acquisition_values is not None, "Acquisition values not found"

                # TODO: Make acq func colours nicer
                acquisition_function_colour = 'grey'

                figure.add_trace(
                    go.Scatter(
                        x=model_prediction.variable_array,
                        y=model_prediction.acquisition_values / acq_func_scaling,
                        line={'color': acquisition_function_colour},
                        name='Acquisition function',
                        legendgroup='acq func',
                        showlegend=True if (row_no == 1 and col_no == 1) else False
                    ),
                    row=row_no, col=col_no
                )

                if model_prediction.modified_acquisition_values is not None:

                    for punish_ind, punished_acq_fun_vals in enumerate(model_prediction.modified_acquisition_values):

                        figure.add_trace(
                            go.Scatter(
                                x=model_prediction.variable_array,
                                y=punished_acq_fun_vals.detach() / acq_func_scaling,
                                line={'color': acquisition_function_colour},
                                name=f'Acq. func., as seen by suggested point {punish_ind + 1}',
                                legendgroup=f'acq func {punish_ind}',
                                showlegend=True if (row_no == 1 and col_no == 1) else False
                            ),
                            row=row_no, col=col_no
                        )

            figure.add_trace(
                go.Scatter(
                    x=variable_values[:, variable_index],
                    y=objective_values[:, objective_index],
                    mode='markers',
                    marker={
                        'color': colour_list_w_opacity,
                        'size': marker_size_list,
                        'line': {
                            'width': 0.0
                        },
                        'opacity': 1.0
                    },
                    marker_symbol=marker_type_list,
                    name='Evaluated points',
                    showlegend=True if (row_no == 1 and col_no == 1) else False,
                    legendgroup='Evaluated points',
                    customdata=np.dstack([list(range(n_evaluated_points)), distance_list])[0],
                    hovertemplate="Param. value: %{x:.3f} <br>"
                                  "Obj. func. value: %{y:.3f} <br>"
                                  "Point number: %{customdata[0]:.0f} <br>"
                                  "Distance to current plane: %{customdata[1]:.3f}"
                ),
                row=row_no,
                col=col_no
            )

            if suggested_points:
                for suggested_point_no, point in enumerate(suggested_points):

                    if suggested_point_no == evaluated_suggested_point_ind:
                        marker_style = 'x'
                        marker_size = 14  # TODO: Write these somewhere general
                    else:
                        marker_style = 'circle'
                        marker_size = 8

                    prediction = point.predicted_objective_values
                    assert prediction is not None, (
                        "Must have calculated predictions for the suggested points before calling this function to "
                        "plot them. (If the model is trained, the optimiser should do this automatically)."
                    )

                    show_legend = False
                    if variable_index == 0 and objective_index == 0 and suggested_point_no == 0:
                        show_legend = True

                    upper_diff = prediction['upper'] - prediction['mean']
                    lower_diff = prediction['mean'] - prediction['lower']

                    figure.add_trace(
                        go.Scatter(
                            x=point.variable_values[variable_index].detach().numpy(),
                            y=prediction['mean'][objective_index].detach().numpy(),
                            error_y={
                                'type': 'data',
                                'symmetric': False,
                                'array': upper_diff[objective_index].detach().numpy(),
                                'arrayminus': lower_diff[objective_index].detach().numpy(),
                                'color': colour_list_suggested_points_w_opacity[suggested_point_no]
                            },
                            mode='markers',
                            marker={
                                'color': colour_list_suggested_points_w_opacity[suggested_point_no],
                                'size': marker_size
                            },
                            marker_symbol=marker_style,
                            name='Suggested points',
                            legendgroup='Suggested points',
                            showlegend=show_legend,
                            customdata=np.dstack([
                                [suggested_point_no],
                                [suggested_point_distance_list[suggested_point_no]],
                                [upper_diff[objective_index].detach().numpy()],
                                [lower_diff[objective_index].detach().numpy()]
                            ])[0],
                            # TODO: Super sweet feature would be to check if upper and lower are equal and then do pm
                            hovertemplate="Param. value: %{x:.3f} <br>"
                                          "Obj. func. value: %{y:.3f}"
                                          " + %{customdata[2]:.3f} /"
                                          " - %{customdata[3]:.3f} <br>"
                                          "Suggested point number: %{customdata[0]:.0f} <br>"
                                          "Distance to current plane: %{customdata[1]:.3f}"
                        ),
                        row=row_no, col=col_no
                    )

            if reference_point is not None and reference_point.objective_values is not None:
                figure.add_trace(
                    go.Scatter(
                        x=[reference_point.variable_values[0, variable_index].item()],
                        y=[reference_point.objective_values[0, objective_index].item()],
                        mode='markers',
                        name='Reference point',
                        visible='legendonly',
                        legendgroup='Reference point',
                        showlegend=True if (row_no == 1 and col_no == 1) else False,
                        marker={
                            'symbol': 'square',
                            'size': 8,
                            'color': 'blue',
                        },
                        hovertemplate="<b>Reference Point</b><br>"
                                      f"{variable_names[variable_index]}:" + " %{x:.3f}<br>"
                                      f"{objective_names[objective_index]}:" + " %{y:.3f}<extra></extra>"
                    ),
                    row=row_no, col=col_no
                )

            figure.update_xaxes(
                range=[model_prediction.variable_array.min(), model_prediction.variable_array.max()],
                row=row_no,
                col=col_no
            )

            if col_no == 1:
                figure.update_yaxes(
                    title_text=objective_names[objective_index],
                    row=row_no,
                    col=col_no
                )

            if row_no == n_objectives:
                figure.update_xaxes(
                    title_text=variable_names[variable_index],
                    row=row_no,
                    col=col_no
                )

    figure.update_layout(
        title={'text': f"Points and predictions{model_prediction.title}"}
    )

    return figure


def _add_labels(
        figure: go.Figure,
        labels_x: list[str],
        labels_y: list[str]
) -> None:

    n_plots = len(labels_x)

    plots_start_middle_end = np.linspace(
        start=-0.07,
        stop=1.07,
        num=n_plots * 2 + 1
    )
    placements = plots_start_middle_end[1::2]

    for label_no, label in enumerate(labels_x):

        figure.add_annotation(
            x=placements[label_no],
            y=-0.09,
            text=label,
            xref="paper",
            yref="paper",
            showarrow=False
        )

    plots_start_middle_end = np.linspace(
        start=-0.12,
        stop=1.12,
        num=n_plots * 2 + 1
    )
    placements = plots_start_middle_end[1::2]

    for label_no, label in enumerate(labels_y):

        figure.add_annotation(
            x=-0.06,
            y=placements[label_no],
            text=label,
            xref="paper",
            yref="paper",
            showarrow=False,
            textangle=-90
        )


def _calculate_grid_model_matrix(
        bounds: torch.Tensor,
        variable_x_index: int,
        variable_y_index: int,
        objective_index: int,
        evaluated_point: torch.Tensor,
        predictor: Predictor,
        normalised: bool,
        n_points_per_dimension: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    variable_tensor_x = torch.linspace(
        start=bounds[0, variable_x_index],
        end=bounds[1, variable_x_index],
        steps=n_points_per_dimension
    )

    variable_tensor_y = torch.linspace(
        start=bounds[0, variable_y_index],
        end=bounds[1, variable_y_index],
        steps=n_points_per_dimension
    )

    grid_x, grid_y = torch.meshgrid(
        variable_tensor_x,
        variable_tensor_y,
        indexing='xy'
    )

    all_variables_tensor = evaluated_point.repeat(
        n_points_per_dimension * n_points_per_dimension,
        1
    )

    for i in range(n_points_per_dimension):
        all_variables_tensor[
            i * n_points_per_dimension:i * n_points_per_dimension + n_points_per_dimension,
            variable_x_index
        ] = grid_x[i, :]

        all_variables_tensor[
            i * n_points_per_dimension:i * n_points_per_dimension + n_points_per_dimension,
            variable_y_index
        ] = grid_y[i, :]

    prediction = predictor.predict_values(
        variable_values=all_variables_tensor,
        normalised=normalised
    )

    prediction_objective_tensor = prediction['mean'][:, objective_index]

    prediction_objective_matrix = prediction_objective_tensor.reshape(n_points_per_dimension, n_points_per_dimension)

    return grid_x, grid_y, prediction_objective_matrix


def _plot_prediction_surface(
        prediction_objective_matrix: torch.Tensor,
        prediction_grid_x: torch.Tensor,
        prediction_grid_y: torch.Tensor,
        point_variable_values: torch.Tensor,
        point_objective_values: torch.Tensor,
        evaluated_point: torch.Tensor,
        evaluated_point_objective_value: float,
        variable_x_index: int,
        variable_y_index: int,
        x_axis_title: str,
        y_axis_title: str,
        z_axis_title: str,
        figure: Optional[go.Figure] = None,
        row_col: Optional[tuple] = None
) -> go.Figure:

    colour_scale = colors.get_colorscale('matter')

    opacity_list, distance_list = opacity_for_multidimensional_points(
        variable_indices=[variable_x_index, variable_y_index],
        variable_values=point_variable_values,
        evaluated_point=evaluated_point,
        alpha_min=0.4,
        alpha_max=1.0
    )

    colour_list = [
        get_continuous_colour(
            colour_scale=colour_scale,
            value=float(1 - distance)
        ) for distance in distance_list
    ]

    colour_list_w_opacity = [
        "rgba(" + colour_list[point_no][4:-1] + f", {opacity_list[point_no]})"
        for point_no in range(len(opacity_list))
    ]

    n_evaluated_points = point_objective_values.shape[DataShape.index_points]

    if figure is None:
        figure = go.Figure()

    if row_col is None:
        current_point_marker_size = 10

        row_col_args = {}
        marker_args = {
            'size': 4
        }
        zaxis_arg = {
            'zaxis': {
                'title': {
                    'text': z_axis_title
                }
            }
        }

    else:

        current_point_marker_size = 4

        row_col_args = {
            'row': row_col[0],
            'col': row_col[1]
        }
        marker_args = {
            'size': 2
        }
        zaxis_arg = {
            'zaxis': {
                'title': {
                    'text': ''
                }
            }
        }

    figure.add_trace(
        go.Surface(
            x=prediction_grid_x.detach().numpy(),
            y=prediction_grid_y.detach().numpy(),
            z=prediction_objective_matrix.detach().numpy(),
            colorscale='deep',
            opacity=1.0,  # Would actually like this at 0.9 but points seem to become 100% visible through at the moment
            name='',  # "Mean model prediction",
            showscale=False,
            hovertemplate=f"{x_axis_title}: " + "%{x:.3f} <br>"
                      f"{y_axis_title}: " + "%{y:.3f} <br>"
                      f"{z_axis_title}: " + "%{z:.3f} <br>"
        ),
        **row_col_args
    )

    figure.add_trace(
        go.Scatter3d(
            x=point_variable_values[:, variable_x_index].detach().numpy(),
            y=point_variable_values[:, variable_y_index].detach().numpy(),
            z=point_objective_values.detach().numpy(),
            mode='markers',
            marker={
                'color': colour_list_w_opacity,
                'opacity': 1.0,
                **marker_args
            },
            name='',  # "Evaluated points",  # Removed name for the hover
            showlegend=False,
            customdata=np.dstack([list(range(n_evaluated_points)), distance_list])[0],
            hovertemplate=f"{x_axis_title}: " + "%{x:.3f} <br>"
                      f"{y_axis_title}: " + "%{y:.3f} <br>"
                      f"{z_axis_title}: " + "%{z:.3f} <br>"
                      "Point number: %{customdata[0]:.0f} <br>"
                      "Distance to current plane: %{customdata[1]:.3f}"
        ),
        **row_col_args
    )

    figure.add_trace(
        go.Scatter3d(
            x=[float(evaluated_point[0, variable_x_index])],
            y=[float(evaluated_point[0, variable_y_index])],
            z=[evaluated_point_objective_value],
            mode='markers',
            marker={
                'color': 'black',
                'opacity': 1.0,
                'size': current_point_marker_size,
                'symbol': 'x'
            },
            name='',  # "Current point",  # Removed name for the hover
            showlegend=False,
            hovertemplate=f"{x_axis_title}: " + "%{x:.3f} <br>"
                      f"{y_axis_title}: " + "%{y:.3f} <br>"
                      f"{z_axis_title}: " + "%{z:.3f} <br>"
                      "This is the current point"
        ),
        **row_col_args
    )

    figure.update_scenes(
        xaxis={
            'title': {
                'text': x_axis_title
            }
        },
        yaxis={
            'title': {
                'text': y_axis_title
            }
        },
        **zaxis_arg,
        **row_col_args
    )

    return figure
