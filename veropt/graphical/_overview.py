from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.express import colors
from plotly.subplots import make_subplots

from veropt.optimiser.optimiser_utility import ReferencePoint
from veropt.optimiser.utility import DataShape


def _plot_point_overview(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        objective_names: list[str],
        variable_names: list[str],
        shown_indices: Optional[list[int]] = None
) -> go.Figure:
    # TODO: Maybe want a longer colour scale to avoid duplicate colours...?
    color_scale = colors.qualitative.T10
    color_scale = colors.convert_colors_to_same_type(color_scale, colortype="rgb")[0]
    n_colors = len(color_scale)

    # TODO: Cool hover shit?
    #   - Even without a dash app, we could add the "sum score" for each point on hover

    n_points = variable_values.shape[0]

    opacity_lines = 0.2

    figure = make_subplots(rows=2, cols=1)

    # TODO: Give the point numbers of all evaluated points (unless it's suggested points?)
    for point_no in range(n_points):

        if shown_indices is not None:
            if point_no not in shown_indices:
                args = {'visible': 'legendonly'}
            else:
                args = {}
        else:
            args = {}

        figure.add_trace(
            go.Scatter(
                x=variable_names,
                y=variable_values[point_no].detach().numpy(),
                name=f"Point no. {point_no}",  # This is currently out of the ones plotted, consider that
                line={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + f", {opacity_lines})"},
                marker={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + ", 1.0)"},
                mode='lines+markers',
                legendgroup=point_no,
                **args
            ),
            row=1,
            col=1
        )

        figure.add_trace(
            go.Scatter(
                x=objective_names,
                y=objective_values[point_no].detach().numpy(),
                line={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + f", {opacity_lines})"},
                marker={'color': "rgba(" + color_scale[point_no % n_colors][4:-1] + ", 1.0)"},
                name=f"Point no. {point_no}",
                mode='lines+markers',
                legendgroup=point_no,
                showlegend=False,
                **args
            ),
            row=2,
            col=1
        )

    figure.update_layout(
        # title={'text': "Plot Title"},
        # xaxis={'title': {'text': "Parameter Number"}},  # Maybe obvious and unnecessary?
        yaxis={'title': {'text': "Parameter Values"}},  # TODO: Add if they're normalised or not
        # TODO: Add if they're predicted or evaluated
        yaxis2={'title': {'text': "Objective Values"}},  # TODO: Add if they're normalised or not
    )

    if n_points < 7:
        figure.update_layout(hovermode="x")

    return figure


def plot_point_overview_separate_subplots(
        variable_values: torch.Tensor,
        objective_values: torch.Tensor,
        objective_names: list[str],
        variable_names: list[str],
        bounds: torch.Tensor,
        shown_indices: Optional[list[int]] = None,
        reference_point: Optional[ReferencePoint] = None
) -> go.Figure:

    # TODO: Maybe want a longer colour scale to avoid duplicate colours...?
    color_scale = colors.qualitative.T10
    color_scale = colors.convert_colors_to_same_type(color_scale, colortype="rgb")[0]
    n_colors = len(color_scale)

    # TODO: Cool hover shit?
    #   - Even without a dash app, we could add the "sum score" for each point on hover

    n_points = variable_values.shape[0]

    n_variables = len(variable_names)
    n_objectives = len(objective_names)

    figure = make_subplots(
        rows=n_objectives,
        cols=n_variables
    )

    # TODO: Give the point numbers of all evaluated points (unless it's suggested points?)
    for point_no in range(n_points):

        if shown_indices is not None:
            if point_no not in shown_indices:
                args = {'visible': 'legendonly'}
            else:
                args = {}
        else:
            args = {}

        for variable_index in range(n_variables):

            for objective_index in range(n_objectives):

                row_no = n_objectives - objective_index
                col_no = variable_index + 1

                figure.add_trace(
                    go.Scatter(
                        x=variable_values[point_no, variable_index].detach().numpy(),
                        y=objective_values[point_no, objective_index].detach().numpy(),
                        mode='markers',
                        marker={
                            'color': color_scale[point_no % n_colors],
                            'line': {
                                'width': 0.0
                            },
                            'opacity': 1.0
                        },
                        name=f"Point no {point_no}",
                        showlegend=True if (row_no == 1 and col_no == 1) else False,
                        legendgroup=f"Point no {point_no}",
                        **args
                    ),
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

    if reference_point is not None:
        for variable_index in range(n_variables):
            for objective_index in range(n_objectives):
                row_no = n_objectives - objective_index
                col_no = variable_index + 1

                if reference_point.objective_values is None:
                    continue

                figure.add_trace(
                    go.Scatter(
                        x=[reference_point.variable_values[0, variable_index].item()],
                        y=[reference_point.objective_values[0, objective_index].item()],
                        mode='markers',
                        marker={
                            'color': 'black',
                            'symbol': 'square'
                        },
                        name="Reference point",
                        showlegend=True if (row_no == 1 and col_no == 1) else False,
                        legendgroup="Reference point",
                        hovertemplate="<b>Reference Point</b><br>"
                                      f"{variable_names[variable_index]}:" + " %{x:.3f}<br>"
                                      f"{objective_names[objective_index]}:" + " %{y:.3f}<extra></extra>"
                    ),
                    row=row_no, col=col_no
                )

    for variable_index in range(n_variables):
        figure.update_xaxes(
            range=[bounds[0, variable_index], bounds[1, variable_index]],
            col=variable_index + 1
        )

    # Note: Currently would merge all the obj. axes
    # for data in figure.data:
    #     data.yaxis = 'y'
    #
    # figure.update_layout(
    #     hoversubplots="axis",
    #     hovermode='y unified',
    # )

    return figure


def _plot_progression(
        objective_values: torch.Tensor,
        objective_names: list[str],
        n_initial_points: int,
        reference_point: Optional[ReferencePoint] = None
) -> go.Figure:

    n_evaluated_points = objective_values.shape[DataShape.index_points]
    n_objectives = objective_values.shape[DataShape.index_dimensions]

    colour_scale = colors.qualitative.Dark2
    if n_objectives > 1:
        colour_list = colors.sample_colorscale(
            colorscale=colour_scale,
            samplepoints=n_objectives
        )
    else:
        colour_list = [colour_scale[0]]

    layout = dict(
        hoversubplots="axis",
        hovermode="x",
        grid=dict(rows=n_objectives, columns=1),
    )

    data = []
    yaxis_names = [f'y{objective_index + 1}' for objective_index in range(n_objectives)]
    yaxis_names[0] = 'y'
    yaxis_names_ver_2 = [f'yaxis{objective_index + 1}' for objective_index in range(n_objectives)]
    yaxis_names_ver_2[0] = 'yaxis'

    for objective_index in range(n_objectives):

        data.append(go.Scatter(
            x=np.arange(n_initial_points),
            y=objective_values.detach().numpy()[:n_initial_points, objective_index],
            name=f"Initial points, objective '{objective_names[objective_index]}'",
            mode='markers',
            marker={
                'symbol': 'diamond',
                'color': colour_list[objective_index],
            },
            xaxis='x',
            yaxis=yaxis_names[objective_index]
        ))

        data.append(go.Scatter(
            x=np.arange(n_initial_points, n_evaluated_points),
            y=objective_values.detach().numpy()[n_initial_points:, objective_index],
            name=f"Bayesian points, objective '{objective_names[objective_index]}'",
            mode='markers',
            marker={
                'color': colour_list[objective_index],
            },
            xaxis='x',
            yaxis=yaxis_names[objective_index]
        ))

        if reference_point is not None and reference_point.objective_values is not None:
            reference_value = reference_point.objective_values[0, objective_index]

            # Horizontal dashed line across the x-axis range
            data.append(go.Scatter(
                x=[-5, n_evaluated_points + 6],
                y=[reference_value, reference_value],
                mode='lines',
                name='Reference value',
                legendgroup='Reference point',
                showlegend=True if objective_index == 0 else False,
                opacity=0.8,
                line={
                    'color': 'black',
                    'dash': 'dash',
                    'width': 2
                },
                xaxis='x',
                yaxis=yaxis_names[objective_index],
                hovertemplate=f"Reference: {reference_value:.3f}<extra></extra>"
            ))

    figure = go.Figure(
        data=data,
        layout=layout
    )

    figure.update_layout(
        xaxis={
            'title': {'text': "Evaluated points"},  # TODO: Add if they're normalised or not
            'range': [-1, n_evaluated_points]
        },
        **{
            yaxis_names_ver_2[objective_index]: {'title': {'text': objective_names[objective_index]}}
            for objective_index in range(n_objectives)
        }
    )

    return figure
