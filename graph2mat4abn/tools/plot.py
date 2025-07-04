from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def plot_error_matrices_big(true_matrix, predicted_matrix, matrix_label=None, figure_title=None, predicted_matrix_text=None, filepath=None, force_max_colorbar_abs_error=None):
    """Interactive Plotly visualization of error matrices."""

    # === Error matrices computation ===
    absolute_error_matrix = true_matrix - predicted_matrix
    epsilon = 0.001
    relative_error_matrix = absolute_error_matrix / (true_matrix + epsilon)*100

    # === Colorbar limits ===
    vmin = np.min([np.min(true_matrix), np.min(predicted_matrix)])
    vmax = np.max([np.max(true_matrix), np.max(predicted_matrix)])
    lim_data = max(abs(vmin), abs(vmax))

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
        "Absolute error (A-B)",
        f"Relative error (A-B)/(A+{epsilon})"
    ]
    cbar_titles = ["eV", "eV", "eV", "%"]

    # === Figure ===
    cbar_y_positions = [1, 1, 1, 1]
    matrices = [true_matrix, predicted_matrix, absolute_error_matrix, relative_error_matrix]

    fig = make_subplots(
        rows=4, cols=1,
        # subplot_titles=titles,
        # horizontal_spacing=0.15,
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
            # colorbar_x = 1,
            # colorbar_y = i
        )
        fig.add_trace(heatmap, row=row, col=col)

    # === Text annotations ===

    # Text under predicted matrix
    if predicted_matrix_text:
        fig.add_annotation(
            text=predicted_matrix_text,
            xref='x2 domain', yref='y2 domain',
            x=1, y=-0.15,
            showarrow=False,
            font=dict(size=12),
            align='right'
        )

    # Absolute error stats
    max_absolute_error = np.max(absolute_error_matrix)
    min_absolute_error = np.min(absolute_error_matrix)
    max_abs = np.max(np.absolute([max_absolute_error, min_absolute_error]))
    fig.add_annotation(
        text=f"max = {max_absolute_error:.2f} eV,  min = {min_absolute_error:.2f} eV,  |max| = {max_abs:.2f} eV",
        xref='x3 domain', yref='y3 domain',
        x=0.5, y=-0.12,
        showarrow=False,
        font=dict(size=12, weight=400),
        align='center'
    )

    # Relative error stats
    max_relative_error = np.max(relative_error_matrix)
    min_relative_error = np.min(relative_error_matrix)
    fig.add_annotation(
        text=f"max = {max_relative_error:.2f}%,  min = {min_relative_error:.2f}%",
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
        filepath = Path(filepath)
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath))
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    else:
        fig.show()


def plot_error_matrices_small(true_matrix, predicted_matrix, matrix_label=None, figure_title=None, predicted_matrix_text=None, filepath=None, n_atoms=None, force_max_colorbar_abs_error=None):
    """Interactive Plotly visualization of error matrices."""

    # === Error matrices computation ===
    absolute_error_matrix = true_matrix - predicted_matrix
    epsilon = 0.001
    relative_error_matrix = absolute_error_matrix / (true_matrix + epsilon)*100

    # === Colorbar limits ===
    vmin = np.min([np.min(true_matrix), np.min(predicted_matrix)])
    vmax = np.max([np.max(true_matrix), np.max(predicted_matrix)])
    lim_data = max(abs(vmin), abs(vmax))

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
        "Absolute error (A-B)",
        f"Relative error (A-B)/(A+{epsilon})"
    ]
    cbar_titles = ["eV", "eV", "eV", "%"]

    # === Figure ===
    cbar_positions = [0.44, 1, 0.44, 1]
    matrices = [true_matrix, predicted_matrix, absolute_error_matrix, relative_error_matrix]

    fig = make_subplots(
        rows=2, cols=2,
        # subplot_titles=titles,
        horizontal_spacing=0.15,
        vertical_spacing=0.17
    )

    for i, matrix in enumerate(matrices):
        row = i // 2 + 1
        col = i % 2 + 1

        heatmap = go.Heatmap(
            z=matrix,
            colorscale='RdYlBu',
            zmin=-cbar_limits[i],
            zmax=cbar_limits[i],
            colorbar=dict(title=cbar_titles[i], len=0.475, yanchor="middle", y=0.807 - 0.585*(row-1)),
            colorbar_x = cbar_positions[i]
        )
        fig.add_trace(heatmap, row=row, col=col)

    # === Subplot titles ===
    fig.update_layout(
        xaxis1=dict(side="top", title_text=titles[0]), yaxis1=dict(autorange="reversed"),
        xaxis2=dict(side="top", title_text=titles[1]), yaxis2=dict(autorange="reversed"),
        xaxis3=dict(side="top", title_text=titles[2]), yaxis3=dict(autorange="reversed"),
        xaxis4=dict(side="top", title_text=titles[3]), yaxis4=dict(autorange="reversed"),
        margin={"l":0,
                "r":0,
                "t":0,
                "b":0}
    )

    # # === Atomic orbitals blocks grid ===
    # if n_atoms is not None:
    #     n_orbitals = 13
    #     minor_ticks = np.arange(-0.5, n_orbitals * n_atoms, n_orbitals)

    #     for i, matrix in enumerate(matrices):
    #         row = i // 2 + 1
    #         col = i % 2 + 1  # Ensure shapes are added to the correct subplot

    #         grid_lines = [
    #             # Vertical grid lines
    #             dict(type="line", x0=x, x1=x, y0=-0.5, y1=n_orbitals * n_atoms - 0.5, line=dict(color="black", width=1))
    #             for x in minor_ticks
    #         ] + [
    #             # Horizontal grid lines
    #             dict(type="line", y0=y, y1=y, x0=-0.5, x1=n_orbitals * n_atoms - 0.5, line=dict(color="black", width=1))
    #             for y in minor_ticks
    #         ]

    #         # Add each grid line to the corresponding subplot
    #         for line in grid_lines:
    #             fig.add_shape(line, row=row, col=col)

    # === Text annotations ===

    # Text under predicted matrix
    if predicted_matrix_text:
        fig.add_annotation(
            text=predicted_matrix_text,
            xref='x2 domain', yref='y2 domain',
            x=1.1, y=-0.15,
            showarrow=False,
            font=dict(size=12),
            align='right'
        )

    # Absolute error stats
    max_absolute_error = np.max(absolute_error_matrix)
    min_absolute_error = np.min(absolute_error_matrix)
    fig.add_annotation(
        text=f"max = {max_absolute_error:.2f} eV, min = {min_absolute_error:.2f} eV",
        xref='x3 domain', yref='y3 domain',
        x=0.5, y=-0.07,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )

    # Relative error stats
    max_relative_error = np.max(relative_error_matrix)
    min_relative_error = np.min(relative_error_matrix)
    fig.add_annotation(
        text=f"max = {max_relative_error:.2f}%, min = {min_relative_error:.2f}%",
        xref='x4 domain', yref='y4 domain',
        x=0.5, y=-0.07,
        showarrow=False,
        font=dict(size=12),
        align='center'
    )

    # === Layout of the whole figure ===
    fig.update_layout(
        height=850,
        width=800,
        title_text=figure_title if figure_title else "Matrix Comparison and Errors",
        title_x=0.5,
        title_y=0.99,
        margin=dict(t=100, b=20)
    )

    # === Output ===
    if filepath:
        filepath = Path(filepath)
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath))
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    else:
        fig.show()


def plot_columns_of_2darray(array_pred, array_true=None, x=None, titles_pred=None, titles_true=None, xlabel=None, ylabel=None, title=None, filepath=None):
    """
    Plot each column of a 2D numpy array as separate traces.
    
    Parameters:
    - array_pred: 2D numpy array to plot (each column will be a separate trace)
    - array_true: Optional true values to plot alongside predictions
    - x: Optional x-axis values (if None, uses array indices)
    - titles_pred: Optional list of names for each prediction column/trace
    - titles_true: Optional list of names for each true column/trace
    - xlabel: Label for x-axis
    - ylabel: Label for y-axis
    - title: Title for the overall plot
    - filepath: Optional path to save the plot (supports .html or .png)
    """
    if array_pred.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")
        
    num_cols = array_pred.shape[1]
    
    # Default color sequence (you can customize this)
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    
    if titles_pred is None:
        titles_pred = [f'Pred Column {i+1}' for i in range(num_cols)]
    elif len(titles_pred) != num_cols:
        raise ValueError("Number of titles_pred must match number of columns")
    
    if titles_true is None:
        titles_true = [f'True Column {i+1}' for i in range(num_cols)]
    elif len(titles_true) != num_cols:
        raise ValueError("Number of titles_true must match number of columns")
    
    if x is None:
        x = np.arange(array_pred.shape[0])
    elif len(x) != array_pred.shape[0]:
        raise ValueError("Length of x must match number of rows in array")
    
    fig = go.Figure()
    
    for col in range(num_cols):
        # Get color for this column (repeats if more columns than colors)
        color = colors[col % len(colors)]
        
        # Add predicted values as dashed line (initially hidden)
        fig.add_trace(go.Scatter(
            x=x,
            y=array_pred[:, col],
            mode='lines',
            name=titles_pred[col],
            line=dict(color=color, dash='dash'),
            opacity=0.8,
            visible='legendonly',  # This makes it hidden by default
            legendgroup=f'group{col}',  # Group pred and true together
            showlegend=True
        ))

        if array_true is not None:
            # Add true values as solid line with same color (initially hidden)
            fig.add_trace(go.Scatter(
                x=x,
                y=array_true[:, col],
                mode='lines',
                name=titles_true[col],
                line=dict(color=color),
                opacity=0.8,
                visible='legendonly',  # This makes it hidden by default
                legendgroup=f'group{col}',  # Group pred and true together
                showlegend=True
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        hovermode='x unified',
        height=800,
        xaxis_title_standoff=15,
    )

    fig.update_xaxes(
        showticklabels=False,
    )

    # === Output ===
    if filepath:
        filepath = Path(filepath)
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath))
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    else:
        fig.show()
    
    return fig