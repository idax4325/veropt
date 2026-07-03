"""Visual confirmation of posterior variance behaviour with and without noise.

NOT part of the pytest suite (file does not start with 'test_').
Run directly:
    python tests/visualise_noise_posterior.py

Layout: 1 row × 2 columns
  Left  — noiseless GP
  Right — noisy GP with noise_std pinned to exactly the same value used when sampling
           the training observations — so model noise = data noise by construction.

Uncertainty is shown the same way as veropt's own prediction plots:
  invisible upper-bound line → lower-bound line filled to upper → mean line on top.
  Training observations carry ±1σ error bars matching the pinned noise.
"""
import argparse

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from veropt.optimiser.kernels import MaternKernel
from veropt.optimiser.model import GPyTorchFullModel, AdamModelOptimiser


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_TRAIN = 12
NOISE_STD = 0.15   # applied both to observations and given to the GP

FILL_COLOUR = 'rgba(180, 180, 180, 0.35)'
MEAN_NOISELESS = 'rgba(31,  119, 180, 1.0)'
MEAN_NOISY = 'rgba(214,  39,  40, 1.0)'
C_TRUE = 'rgba(80,   80,  80, 0.5)'
C_TRAIN = 'rgba(30,   30,  30, 0.85)'
C_VLINE = 'rgba(160, 160, 160, 0.45)'


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def make_training_data(
        n_train: int,
        noise_std: float,
        seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (x_train [N,1], y_noisy [N,1], y_clean [N]) for a noisy sine."""
    torch.manual_seed(seed)
    x_train = torch.linspace(0.0, 1.0, n_train).unsqueeze(-1)
    y_clean = torch.sin(x_train * 2 * torch.pi).squeeze(-1)
    y_noisy = y_clean + torch.randn(n_train) * noise_std
    return x_train, y_noisy.unsqueeze(-1), y_clean


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_and_train_model(
        x_train: torch.Tensor,
        y_targets: torch.Tensor,
        noise_std_value: float | None,
        max_iter: int = 200,
) -> GPyTorchFullModel:
    """Build, train and return a 1-D GP.  Pin noise if noise_std_value is given."""
    kernel = MaternKernel(n_variables=1)
    model = GPyTorchFullModel.from_the_beginning(
        n_variables=1,
        n_objectives=1,
        single_model_list=[kernel],
        model_optimiser=AdamModelOptimiser(),
        max_iter=max_iter,
        verbose=False
    )
    noise_tensor = torch.tensor([noise_std_value]) if noise_std_value is not None else None
    model.train_model(
        variable_values=x_train,
        objective_values=y_targets,
        noise_std_in_model_space=noise_tensor
    )
    return model


def get_posterior(
        model: GPyTorchFullModel,
        x: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, upper +2σ, lower −2σ) numpy arrays at x."""
    gpytorch_model = model._model_list[0].model_with_data
    assert gpytorch_model is not None
    gpytorch_model.eval()
    with torch.no_grad():
        posterior = gpytorch_model.likelihood(gpytorch_model(x))
    mean = posterior.mean.squeeze().numpy()
    std = posterior.variance.squeeze().sqrt().numpy()
    return mean, mean + std, mean - std


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def add_gp_band(
        fig: go.Figure,
        col: int,
        x: np.ndarray,
        mean: np.ndarray,
        upper: np.ndarray,
        lower: np.ndarray,
        mean_colour: str,
        label: str,
) -> None:
    """Upper → lower (filled) → mean — same trace stack as veropt's prediction plots."""
    fig.add_trace(
        go.Scatter(
            x=x, y=upper,
            line=dict(width=0.0, color=FILL_COLOUR),
            name=f"{label} upper",
            legendgroup=label,
            showlegend=False,
        ),
        row=1, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=lower,
            fill='tonexty',
            fillcolor=FILL_COLOUR,
            line=dict(width=0.0, color=FILL_COLOUR),
            name=f"{label} ±1σ",
            legendgroup=label,
            showlegend=True,
        ),
        row=1, col=col,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=mean,
            mode='lines',
            line=dict(color=mean_colour, width=2),
            name=f"{label} mean",
            legendgroup=label,
            showlegend=True,
        ),
        row=1, col=col,
    )


def add_true_function(fig: go.Figure, x: np.ndarray, y: np.ndarray, col: int) -> None:
    fig.add_trace(
        go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(color=C_TRUE, dash='dot', width=1.5),
            name="True function",
            legendgroup="true",
            showlegend=(col == 1),
        ),
        row=1, col=col,
    )


def add_observations(
        fig: go.Figure,
        x_train: np.ndarray,
        y_train: np.ndarray,
        noise_std: float | None,
        col: int,
) -> None:
    """Plot training observations with optional ±1σ error bars."""
    error_y = (
        dict(type='constant', value=noise_std, color=C_TRAIN, thickness=1.5, width=5)
        if noise_std is not None
        else None
    )
    fig.add_trace(
        go.Scatter(
            x=x_train, y=y_train,
            mode='markers',
            marker=dict(
                symbol='circle',
                size=8,
                color=C_TRAIN,
                line=dict(width=1.5, color='white'),
            ),
            error_y=error_y,
            name="Observations" + (f" (±{noise_std} σ)" if noise_std else ""),
            legendgroup="train",
            showlegend=(col == 1),
        ),
        row=1, col=col,
    )


def add_training_vlines(fig: go.Figure, x_values: np.ndarray) -> None:
    for x_val in x_values:
        for col in (1, 2):
            fig.add_vline(
                x=float(x_val),
                line=dict(color=C_VLINE, width=0.8, dash='dot'),
                row=1, col=col,
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_figure(noise_std: float, n_train: int) -> go.Figure:
    x_train, y_targets, y_clean = make_training_data(n_train=n_train, noise_std=noise_std)
    x_grid = torch.linspace(-0.05, 1.05, 400).unsqueeze(-1)

    print("Training noiseless model ...")
    model_noiseless = build_and_train_model(x_train, y_targets, noise_std_value=None)

    print(f"Training noisy model (noise_std={noise_std}) ...")
    model_noisy = build_and_train_model(x_train, y_targets, noise_std_value=noise_std)

    x_np = x_grid.squeeze().numpy()
    x_train_np = x_train.squeeze().numpy()
    y_train_np = y_targets.squeeze().numpy()
    y_true_np = torch.sin(torch.tensor(x_np) * 2 * torch.pi).numpy()

    mean_nl, upper_nl, lower_nl = get_posterior(model_noiseless, x_grid)
    mean_n, upper_n, lower_n = get_posterior(model_noisy, x_grid)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Noiseless GP — variance collapses at training points",
            f"Noisy GP (noise_std = {noise_std}) — variance stays at noise floor",
        ],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )

    add_gp_band(fig, col=1, x=x_np, mean=mean_nl, upper=upper_nl, lower=lower_nl,
                mean_colour=MEAN_NOISELESS, label="Noiseless")
    add_gp_band(fig, col=2, x=x_np, mean=mean_n, upper=upper_n, lower=lower_n,
                mean_colour=MEAN_NOISY, label="Noisy")

    for col in (1, 2):
        add_true_function(fig, x=x_np, y=y_true_np, col=col)

    # Noiseless panel: no error bars (no noise assumed)
    add_observations(fig, x_train=x_train_np, y_train=y_train_np, noise_std=None, col=1)
    # Noisy panel: error bars at ±noise_std
    add_observations(fig, x_train=x_train_np, y_train=y_train_np, noise_std=noise_std, col=2)

    add_training_vlines(fig, x_train_np)

    fig.update_layout(
        title=dict(
            text=(
                f"GP posterior — noiseless vs. pinned noise  (noise_std = {noise_std:.2f})<br>"
                "<sup>Shaded band = ±1σ likelihood posterior  ·  "
                "Observations sampled with the same noise_std given to the model</sup>"
            ),
            x=0.5,
        ),
        height=500,
        width=1250,
        legend=dict(orientation='h', y=-0.12, x=0.5, xanchor='center'),
    )
    fig.update_xaxes(title_text="x", range=[-0.05, 1.05])
    fig.update_yaxes(title_text="y", col=1)

    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualise GP posterior with and without noise.")
    parser.add_argument('--save-html', action='store_true', help="Write a standalone HTML file next to this script.")
    args = parser.parse_args()

    fig = build_figure(noise_std=NOISE_STD, n_train=N_TRAIN)

    if args.save_html:
        html_path = __file__.replace('.py', '.html')
        fig.write_html(html_path)
        print(f"Plot saved to: {html_path}")

    fig.show()


if __name__ == '__main__':
    main()
