import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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