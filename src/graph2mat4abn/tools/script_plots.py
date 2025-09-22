import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import plotly.graph_objects as go
from pathlib import Path

def plot_grad_norms(grad_stats):
    """
    Plot per–parameter gradient-norm curves across epochs.

    Parameters
    ----------
    grad_stats : list[dict[str, float]]
        Each element is the *dict* you stored at one epoch:
        ``{"layer.weight": 0.23, "layer.bias": 0.04, ...}``.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive figure (also written to *grad_norms.html* in the CWD).
    """
    # ------------------------------------------------------------------ #
    # 1. Collect every parameter name that ever appeared
    # ------------------------------------------------------------------ #
    all_params = sorted({name for d in grad_stats for name in d})

    # ------------------------------------------------------------------ #
    # 2. Build the per-parameter time-series (None for missing epochs)
    # ------------------------------------------------------------------ #
    series = {p: [] for p in all_params}
    for d in grad_stats:                       # epoch loop
        for p in all_params:
            series[p].append(d.get(p, None))   # None = break in line

    epochs = list(range(len(grad_stats)))

    # ------------------------------------------------------------------ #
    # 3. Plot
    # ------------------------------------------------------------------ #
    fig = go.Figure()
    for p in all_params:
        fig.add_trace(
            go.Scatter(
                x=epochs,
                y=series[p],
                mode="lines",
                name=p,
                connectgaps=True,          # ignore missing steps gracefully
            )
        )

    fig.update_layout(
        title="Gradient L2-norms per parameter",
        xaxis_title="Epoch",
        yaxis_title="‖∇θ‖₂",
        yaxis_type="log",                  # norms often span several orders
        legend=dict(title="Parameter"),
    )

    # Save and return
    # out = Path("grad_norms.html")
    # fig.write_html(out)
    # print(f"Gradient plot written to {out.resolve()}")
    return fig


import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
from typing import Dict, List, Sequence, Union, Optional


def update_loss_plots(history, verbose=False):
    def detach_list(tensor_list):
        return [
            t.detach().item() if isinstance(t, torch.Tensor)
            else t if t is None
            else float(t)
            for t in tensor_list
        ]
    
    # Prepare data

    if not history["val_loss_extra"]:
        df = pd.DataFrame(
            np.array([
                detach_list(history["train_loss"]),
                detach_list(history["val_loss"]),
                detach_list(history["train_edge_loss"]),
                detach_list(history["val_edge_loss"]),
                detach_list(history["train_node_loss"]),
                detach_list(history["val_node_loss"]),
                detach_list(history["learning_rate"]),
            ]).T,
            columns=["Train total", "Val total", "Train edge", "Val edge", "Train node", "Val node", "Learning rate"],
        )
    else:
        df = pd.DataFrame(
            np.array([
                detach_list(history["train_loss"]),
                detach_list(history["val_loss"]),
                detach_list(history["val_loss_extra"]),
                detach_list(history["train_edge_loss"]),
                detach_list(history["val_edge_loss"]),
                detach_list(history["val_edge_loss_extra"]),
                detach_list(history["train_node_loss"]),
                detach_list(history["val_node_loss"]),
                detach_list(history["val_node_loss_extra"]),
                detach_list(history["learning_rate"]),
            ]).T,
            columns=["Train total", "Val total", "Val_extra total", "Train edge", "Val edge", "Val_extra edge", "Train node", "Val node", "Val_extra node", "Learning rate"],
        )

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add loss traces (primary y-axis)
    loss_colors = {
        "Train total": "blue",
        "Val total": "red",
        "Val_extra total": "magenta",
        "Train edge": "blue",
        "Val edge": "red",
        "Val_extra edge": "magenta",
        "Train node": "blue",
        "Val node": "red",
        "Val_extra node": "magenta",
    }
    loss_dashes = {
        "Train total": "solid",
        "Val total": "solid",
        "Val_extra total": "solid",
        "Train edge": "dash",
        "Val edge": "dash",
        "Val_extra edge": "dash",
        "Train node": "dot",
        "Val node": "dot",
        "Val_extra node": "dot"
    }
    
    for col in df.columns[:-1]:  # All columns except Learning rate
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                name=col,
                line=dict(color=loss_colors[col], dash=loss_dashes[col]),
                legendgroup=col.split()[1] if col.split()[0] in ["Train", "Val"] else col,
                connectgaps=True
            ),
            secondary_y=False
        )
    
    # Add learning rate trace (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Learning rate"],
            name="Learning rate",
            line=dict(color="lightgreen", dash="solid"),
            legendgroup="Learning rate"
        ),
        secondary_y=True
    )
    
    # Set axis titles and layout
    loss_values = df.drop(columns=["Learning rate"]).to_numpy().flatten()
    loss_values = [v for v in loss_values if v is not None]

    ylim_up = int(np.percentile(loss_values, 95)) + 10
    fig.update_layout(
        title=f"Loss curves",
        xaxis_title="Epoch number",
        yaxis=dict(
            title="Loss (eV²)",
            showgrid=True,
            # type="log",
            range=[-5, ylim_up]
        ),
        yaxis2=dict(
            title="Learning rate",
            showgrid=False,
            type="log",
            side="right",
            tickformat=".0e", 
            dtick=1,
        ),
        grid=dict(xside="bottom", yside="left"),
        legend=dict(
            x=1.1,  
            xanchor="left",  
            y=1.0,  
            yanchor="top"  
        ),
        margin=dict(r=150)
    )
    return fig


def plot_hamiltonian(true_matrix, predicted_matrix, matrix_label=None, figure_title=None, predicted_matrix_text=None, filepath=None, force_max_colorbar_abs_error=None):
    """Interactive Plotly visualization of error matrices."""

    # === Error matrices computation ===
    absolute_error_matrix = true_matrix - predicted_matrix

    threshold = 0.001
    mask = true_matrix >= threshold
    relative_error_matrix = np.where(mask, absolute_error_matrix / (true_matrix + threshold) * 100, 0)

    # === Colorbar limits ===
    vmin = np.min([np.min(true_matrix), np.min(predicted_matrix)])
    vmax = np.max([np.max(true_matrix), np.max(predicted_matrix)])
    lim_data = max(np.abs(vmin), np.abs(vmax))

    if force_max_colorbar_abs_error is None:
        lim_abs = np.max(np.abs(absolute_error_matrix))
    else:
        lim_abs = force_max_colorbar_abs_error

    lim_rel = 100.0  # %

    cbar_limits = [lim_data, lim_data, lim_abs, lim_rel]

    # === Titles ===
    if matrix_label is None:
        matrix_label = ''
    titles = [
        "True " + matrix_label,
        "Predicted " + matrix_label,
        "Absolute error (T-P)",
        f"Relative error (T-P)/(T) (masked where T is above {threshold})"
    ]
    cbar_titles = ["eV", "eV", "eV", "%"]

    # === Figure ===
    matrices = [true_matrix, predicted_matrix, absolute_error_matrix, relative_error_matrix]

    fig = make_subplots(
        rows=4, cols=1,
        vertical_spacing=0.1
    )

    for i, matrix in enumerate(matrices):
        row = i + 1
        col = 1

        heatmap = go.Heatmap(
            z=matrix,
            colorscale='RdBu',
            zmin=-cbar_limits[i],
            zmax=cbar_limits[i],
            
            colorbar=dict(title=cbar_titles[i], len=0.21, y=(0.92-0.275*i), ),
        )
        fig.add_trace(heatmap, row=row, col=col)

    # === Text annotations ===

    # Text under predicted matrix
    if predicted_matrix_text is not None:
        fig.add_annotation(
            text=predicted_matrix_text,
            xref='x2 domain', yref='y2 domain',
            x=1, y=-0.15,
            showarrow=False,
            font=dict(size=12),
            align='right'
        )

    # Absolute error stats
    abs = np.abs(absolute_error_matrix)
    mean = np.mean(abs[absolute_error_matrix != 0])
    std = np.std(abs[absolute_error_matrix != 0])

    max_absolute_error = np.max(absolute_error_matrix)
    min_absolute_error = np.min(absolute_error_matrix)
    max_abs = np.max(np.abs([max_absolute_error, min_absolute_error]))

    fig.add_annotation(
        text=f"mean_nnz(|T-P|) = {mean:.3f} eV, std_nnz(|T-P|) = {std:.3f} eV, |max| = {max_abs:.3f} eV",
        xref='x3 domain', yref='y3 domain',
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, weight=400),
        align='center'
    )

    # Relative error stats
    abs = np.abs(relative_error_matrix)
    mean = np.mean(abs[relative_error_matrix != 0])
    std = np.std(abs[relative_error_matrix != 0])

    max_relative_error = np.max(relative_error_matrix)
    min_relative_error = np.min(relative_error_matrix)
    max_abs = np.max(np.abs([max_relative_error, min_relative_error]))

    fig.add_annotation(
        text=f"mean_nnz(|T-P|) = {mean:.3f} %, std_nnz(|T-P|) = {std:.3f} %, |max| = {max_abs:.3f} %",
        xref='x4 domain', yref='y4 domain',
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, weight=400),
        align='center'
    )

    # === Layout of the whole figure ===
    fig.update_layout(
        height=1200,
        width=900,
        title_text=figure_title if figure_title else "Matrix Comparison and Errors",
        title_x=0.46,
        title_y=0.99,
        margin=dict(t=100, b=20, l=30),

        # Subplot titles
        xaxis1=dict(side="top", title_text=titles[0]), yaxis1=dict(autorange="reversed"),
        xaxis2=dict(side="top", title_text=titles[1]), yaxis2=dict(autorange="reversed"),
        xaxis3=dict(side="top", title_text=titles[2]), yaxis3=dict(autorange="reversed"),
        xaxis4=dict(side="top", title_text=titles[3]), yaxis4=dict(autorange="reversed"),
        font=dict(size=12),
    )

    # === Output ===
    if filepath:
        filepath = filepath
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath), height=1200, width=900,)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
        
    else:
        fig.show()

    del fig