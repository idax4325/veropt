from __future__ import annotations

from typing import Literal, Optional, Union

import numpy as np
import torch
from plotly import graph_objects as go
from plotly.express import colors
from plotly.subplots import make_subplots

from veropt.optimiser.optimiser_utility import ReferencePoint, SuggestedPoints
from veropt.optimiser.utility import DataShape

type UncertaintyStyle = Literal['ellipse', 'error_bars']

_ELLIPSE_THETA = np.linspace(0, 2 * np.pi, 60)


def _build_ellipse_traces(
        x_centres: np.ndarray,
        y_centres: np.ndarray,
        sigma_x: float,
        sigma_y: float,
        colour: str,
        name: str,
        show_legend: bool,
        legend_group: str,
) -> go.Scatter:
    """Return a single batched Scatter trace that draws one ellipse per point.

    All ellipses are concatenated with None separators so the entire group is
    one Plotly trace, keeping the legend tidy regardless of point count.
    """
    xs: list[float | None] = []
    ys: list[float | None] = []
    for cx, cy in zip(x_centres, y_centres):
        xs.extend((cx + sigma_x * np.cos(_ELLIPSE_THETA)).tolist() + [None])
        ys.extend((cy + sigma_y * np.sin(_ELLIPSE_THETA)).tolist() + [None])

    return go.Scatter(
        x=xs,
        y=ys,
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),
        fill='toself',
        fillcolor=colour,
        name=name,
        legendgroup=legend_group,
        showlegend=show_legend,
    )


def _plot_pareto_front_grid(
        objective_values: torch.Tensor,
        objective_names: list[str],
        pareto_optimal_indices: list[int],
        n_initial_points: int,
        suggested_points: Optional[SuggestedPoints] = None,
        reference_point: Optional[ReferencePoint] = None,
        noise_std_per_objective: Optional[torch.Tensor] = None,
        uncertainty_style: UncertaintyStyle = 'ellipse',
        return_figure: bool = False
) -> Union[go.Figure, None]:

    n_objectives = len(objective_names)

    figure = make_subplots(
        rows=n_objectives - 1,
        cols=n_objectives - 1
    )

    for objective_index_x in range(n_objectives - 1):
        for objective_index_y in range(1, n_objectives):

            row = objective_index_y
            col = objective_index_x + 1

            if not objective_index_x == objective_index_y:
                figure = _add_pareto_traces_2d(
                    figure=figure,
                    objective_values=objective_values,
                    objective_index_x=objective_index_x,
                    objective_index_y=objective_index_y,
                    objective_names=objective_names,
                    pareto_optimal_indices=pareto_optimal_indices,
                    n_initial_points=n_initial_points,
                    suggested_points=suggested_points,
                    reference_point=reference_point,
                    noise_std_per_objective=noise_std_per_objective,
                    uncertainty_style=uncertainty_style,
                    row=row,
                    col=col
                )

            if col == 1:
                figure.update_yaxes(title_text=objective_names[objective_index_y], row=row, col=col)

            if row == n_objectives - 1:
                figure.update_xaxes(title_text=objective_names[objective_index_x], row=row, col=col)

    if return_figure:
        return figure
    else:
        figure.show()
        return None


def _add_pareto_traces_2d(
        figure: go.Figure,
        objective_values: torch.Tensor,
        objective_index_x: int,
        objective_index_y: int,
        objective_names: list[str],
        pareto_optimal_indices: list[int],
        n_initial_points: int,
        suggested_points: Optional[SuggestedPoints] = None,
        reference_point: Optional[ReferencePoint] = None,
        noise_std_per_objective: Optional[torch.Tensor] = None,
        uncertainty_style: UncertaintyStyle = 'ellipse',
        row: Optional[int] = None,
        col: Optional[int] = None
) -> go.Figure:

    # Note: Must pass all points to this function or point numbers will be wrong

    n_evaluated_points = objective_values.shape[DataShape.index_points]
    point_numbers = np.arange(n_evaluated_points).reshape(n_evaluated_points, 1)

    dominating_objective_values = objective_values[pareto_optimal_indices]

    if row is None and col is None:
        row_col_info: dict = {}
        show_legend = True
    else:
        row_col_info = {'row': row, 'col': col}
        show_legend = (row == 1 and col == 1)

    color_scale = colors.qualitative.Plotly
    color_evaluated_points = color_scale[0]
    color_initial_points = color_scale[2]

    # ------------------------------------------------------------------
    # Noise uncertainty display (ellipses or error bars) on all points
    # ------------------------------------------------------------------

    if noise_std_per_objective is not None:
        sigma_x = float(noise_std_per_objective[objective_index_x])
        sigma_y = float(noise_std_per_objective[objective_index_y])

        if uncertainty_style == 'ellipse':

            def _rgba_with_alpha(hex_or_rgb: str, alpha: float = 0.18) -> str:
                """Convert a Plotly colour string to rgba with the given alpha."""
                import re
                match = re.match(r'rgb\((\d+),\s*(\d+),\s*(\d+)\)', hex_or_rgb)
                if match:
                    r, g, b = match.groups()
                    return f'rgba({r},{g},{b},{alpha})'
                # hex colour
                hex_str = hex_or_rgb.lstrip('#')
                r, g, b = int(hex_str[0:2], 16), int(hex_str[2:4], 16), int(hex_str[4:6], 16)
                return f'rgba({r},{g},{b},{alpha})'

            pareto_set = set(pareto_optimal_indices)
            initial_non_pareto = [idx for idx in range(n_initial_points) if idx not in pareto_set]
            bayesian_non_pareto = [idx for idx in range(n_initial_points, n_evaluated_points) if idx not in pareto_set]

            for group_colour, point_indices, is_first_group in [
                (color_initial_points, initial_non_pareto, True),
                (color_evaluated_points, bayesian_non_pareto, False),
            ]:
                if len(point_indices) == 0:
                    continue
                x_vals = objective_values[point_indices, objective_index_x].numpy()
                y_vals = objective_values[point_indices, objective_index_y].numpy()
                ellipse_fill = _rgba_with_alpha(group_colour, alpha=0.18)
                ellipse_trace = _build_ellipse_traces(
                    x_centres=x_vals,
                    y_centres=y_vals,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    colour=ellipse_fill,
                    name='Noise (±1σ)',
                    show_legend=show_legend and is_first_group,
                    legend_group='noise_ellipses',
                )
                figure.add_trace(ellipse_trace, **row_col_info)

            # Dominating points get black ellipses
            if len(pareto_optimal_indices) > 0:
                dom_x = dominating_objective_values[:, objective_index_x].numpy()
                dom_y = dominating_objective_values[:, objective_index_y].numpy()
                dom_ellipse_trace = _build_ellipse_traces(
                    x_centres=dom_x,
                    y_centres=dom_y,
                    sigma_x=sigma_x,
                    sigma_y=sigma_y,
                    colour='rgba(0,0,0,0.18)',
                    name='Noise (±1σ, dominating)',
                    show_legend=show_legend and len(initial_non_pareto) == 0 and len(bayesian_non_pareto) == 0,
                    legend_group='noise_ellipses',
                )
                figure.add_trace(dom_ellipse_trace, **row_col_info)

    # ------------------------------------------------------------------
    # Standard point scatter traces
    # ------------------------------------------------------------------

    def _make_error(sigma: float) -> Optional[dict]:
        if noise_std_per_objective is not None and uncertainty_style == 'error_bars':
            return dict(type='constant', value=sigma, color='rgba(100,100,100,0.6)', thickness=1.2, width=4)
        return None

    def _hover_template(point_label: str, customdata_index_offset: int = 0) -> str:
        """Build a hovertemplate string, appending ±std when noise is available."""
        pt_col = customdata_index_offset
        x_col = customdata_index_offset + 1
        y_col = customdata_index_offset + 2
        if noise_std_per_objective is not None:
            sx = float(noise_std_per_objective[objective_index_x])
            sy = float(noise_std_per_objective[objective_index_y])
            return (
                f"{point_label}: %{{customdata[{pt_col}]:.0f}} <br>"
                f"{objective_names[objective_index_x]}: %{{customdata[{x_col}]:.3f}} ±{sx:.3f} <br>"
                f"{objective_names[objective_index_y]}: %{{customdata[{y_col}]:.3f}} ±{sy:.3f} <br>"
            )
        else:
            return (
                f"{point_label}: %{{customdata[{pt_col}]:.0f}} <br>"
                f"{objective_names[objective_index_x]}: " + "%{x:.3f} <br>"
                f"{objective_names[objective_index_y]}: " + "%{y:.3f} <br>"
            )

    def _make_customdata(indices_slice: slice | list[int]) -> np.ndarray:
        """Pack point_number, x-value, y-value into customdata columns."""
        pt_nums = point_numbers[indices_slice]
        x_vals = objective_values[indices_slice, objective_index_x].numpy().reshape(-1, 1)
        y_vals = objective_values[indices_slice, objective_index_y].numpy().reshape(-1, 1)
        return np.hstack([pt_nums, x_vals, y_vals])

    figure.add_trace(
        go.Scatter(
            x=objective_values[:n_initial_points, objective_index_x],
            y=objective_values[:n_initial_points, objective_index_y],
            mode='markers',
            name='Initial points',
            legendgroup='Initial points',
            showlegend=show_legend,
            marker={'symbol': 'diamond', 'color': color_scale[2]},
            error_x=(
                _make_error(float(noise_std_per_objective[objective_index_x]))
                if noise_std_per_objective is not None else None
            ),
            error_y=(
                _make_error(float(noise_std_per_objective[objective_index_y]))
                if noise_std_per_objective is not None else None
            ),
            customdata=_make_customdata(slice(None, n_initial_points)),
            hovertemplate=_hover_template("Point number"),
        ),
        **row_col_info
    )

    figure.add_trace(
        go.Scatter(
            x=objective_values[n_initial_points:, objective_index_x],
            y=objective_values[n_initial_points:, objective_index_y],
            mode='markers',
            name='Bayesian points',
            legendgroup='Bayesian points',
            showlegend=show_legend,
            marker={'color': color_evaluated_points},
            error_x=(
                _make_error(float(noise_std_per_objective[objective_index_x]))
                if noise_std_per_objective is not None else None
            ),
            error_y=(
                _make_error(float(noise_std_per_objective[objective_index_y]))
                if noise_std_per_objective is not None else None
            ),
            customdata=_make_customdata(slice(n_initial_points, None)),
            hovertemplate=_hover_template("Point number"),
        ),
        **row_col_info
    )

    marker_type_dominating = np.array(['circle'] * len(pareto_optimal_indices)).astype('U20')
    dominating_initials = n_initial_points > np.array(pareto_optimal_indices)
    marker_type_dominating[dominating_initials] = 'diamond'
    marker_type_dominating_list = marker_type_dominating.tolist()

    figure.add_trace(
        go.Scatter(
            x=dominating_objective_values[:, objective_index_x],
            y=dominating_objective_values[:, objective_index_y],
            mode='markers',
            marker={'color': 'black', 'symbol': marker_type_dominating_list},
            name='Dominating points',
            legendgroup='Dominating points',
            showlegend=show_legend,
            customdata=_make_customdata(pareto_optimal_indices),
            hovertemplate=_hover_template("Point number"),
        ),
        **row_col_info
    )

    if suggested_points is not None:

        suggested_point_color = 'rgb(139, 0, 0)'

        for suggested_point_no, point in enumerate(suggested_points):

            prediction = point.predicted_objective_values

            if prediction:

                upper_diff = prediction['upper'] - prediction['mean']
                lower_diff = prediction['mean'] - prediction['lower']

                figure.add_trace(
                    go.Scatter(
                        x=prediction['mean'][objective_index_x].detach().numpy(),
                        y=prediction['mean'][objective_index_y].detach().numpy(),
                        error_x={
                            'type': 'data',
                            'symmetric': False,
                            'array': upper_diff[objective_index_x].detach().numpy(),
                            'arrayminus': lower_diff[objective_index_x].detach().numpy(),
                            'color': suggested_point_color
                        },
                        error_y={
                            'type': 'data',
                            'symmetric': False,
                            'array': upper_diff[objective_index_y].detach().numpy(),
                            'arrayminus': lower_diff[objective_index_y].detach().numpy(),
                            'color': suggested_point_color
                        },
                        mode='markers',
                        marker={'color': suggested_point_color},
                        name='Suggested points',
                        legendgroup="Suggested points",
                        showlegend=True if (suggested_point_no == 0 and show_legend) else False,
                        visible='legendonly',
                        customdata=np.dstack([
                                    [prediction['lower'][objective_index_x].detach().numpy()],
                                    [prediction['upper'][objective_index_x].detach().numpy()],
                                    [prediction['lower'][objective_index_y].detach().numpy()],
                                    [prediction['upper'][objective_index_y].detach().numpy()],
                        ])[0],
                        hovertemplate=f"Suggested point number: {suggested_point_no} <br>"
                                      f"{objective_names[objective_index_x]}: "
                                      "%{x:.3f} (%{customdata[0]:.3f} to %{customdata[1]:.3f}) <br>"
                                      f"{objective_names[objective_index_y]}: "
                                      "%{y:.3f} (%{customdata[2]:.3f} to %{customdata[3]:.3f}) <br>"
                    ),
                    **row_col_info
                )

    if reference_point is not None and reference_point.objective_values is not None:
        figure.add_trace(
            go.Scatter(
                x=[reference_point.objective_values[0, objective_index_x]],
                y=[reference_point.objective_values[0, objective_index_y]],
                mode='markers',
                name='Reference point',
                legendgroup='Reference point',
                showlegend=show_legend,
                marker={'symbol': 'square', 'color': 'purple'},
                hovertemplate="Reference Point <br>"
                              f"{objective_names[objective_index_x]}:" + " %{x:.3f} <br>"
                              f"{objective_names[objective_index_y]}:" + " %{y:.3f} <br>"
            ),
            **row_col_info
        )

    return figure


def _plot_pareto_front(
        objective_values: torch.Tensor,
        pareto_optimal_indices: list[int],
        plotted_objective_indices: list[int],
        objective_names: list[str],
        n_initial_points: int,
        suggested_points: Optional[SuggestedPoints] = None,
        reference_point: Optional[ReferencePoint] = None,
        noise_std_per_objective: Optional[torch.Tensor] = None,
        uncertainty_style: UncertaintyStyle = 'ellipse',
        return_figure: bool = False
) -> Union[go.Figure, None]:

    if len(plotted_objective_indices) == 2:

        obj_ind_x = plotted_objective_indices[0]
        obj_ind_y = plotted_objective_indices[1]

        figure = go.Figure()

        figure = _add_pareto_traces_2d(
            figure=figure,
            objective_values=objective_values,
            objective_index_x=obj_ind_x,
            objective_index_y=obj_ind_y,
            objective_names=objective_names,
            pareto_optimal_indices=pareto_optimal_indices,
            n_initial_points=n_initial_points,
            suggested_points=suggested_points,
            reference_point=reference_point,
            noise_std_per_objective=noise_std_per_objective,
            uncertainty_style=uncertainty_style,
        )

        figure.update_xaxes(title_text=objective_names[obj_ind_x])
        figure.update_yaxes(title_text=objective_names[obj_ind_y])

    elif len(plotted_objective_indices) == 3:

        # TODO: Add suggested points
        # TODO: Add dominating points

        plotted_obj_vals = objective_values[:, plotted_objective_indices]

        figure = go.Figure(data=[go.Scatter3d(
            x=plotted_obj_vals[:, plotted_objective_indices[0]],
            y=plotted_obj_vals[:, plotted_objective_indices[1]],
            z=plotted_obj_vals[:, plotted_objective_indices[2]],
            mode='markers'
        )])

    else:
        raise ValueError(f"Can plot pareto front of either 2 or 3 objectives, got {len(plotted_objective_indices)}")

    if return_figure:
        return figure

    else:
        figure.show()
        return None
