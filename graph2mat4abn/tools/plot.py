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
    mean = np.mean(absolute_error_matrix)
    std = np.std(absolute_error_matrix)
    fig.add_annotation(
        text=f"max = {max_absolute_error:.2f} eV,  min = {min_absolute_error:.2f} eV,  |max| = {max_abs:.2f} eV,  mean = {mean:.2e}, std = {std:.2e}",
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
            fig.write_image(str(filepath), height=1200, width=900,)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
        
    else:
        fig.show()

    del fig


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
    max_abs = np.max(np.absolute([max_absolute_error, min_absolute_error]))
    mean = np.mean(absolute_error_matrix)
    std = np.std(absolute_error_matrix)
    fig.add_annotation(
        text=f"max = {max_absolute_error:.2f} eV,  min = {min_absolute_error:.2f} eV,  |max| = {max_abs:.2f} eV,  mean = {mean:.2e}, std = {std:.2e}",
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
        
        # Add predicted values as dashed line 
        fig.add_trace(go.Scatter(
            x=x,
            y=array_pred[:, col],
            mode='lines',
            name=titles_pred[col],
            line=dict(color=color, dash='dash'),
            opacity=0.8,
            # visible='legendonly',  # This makes it hidden by default
            legendgroup=f'group{col}',  # Group pred and true together
            showlegend=True
        ))

        if array_true is not None:
            # Add true values as solid line with same color 
            fig.add_trace(go.Scatter(
                x=x,
                y=array_true[:, col],
                mode='lines',
                name=titles_true[col],
                line=dict(color=color),
                opacity=0.8,
                # visible='legendonly',  # This makes it hidden by default
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



def plot_predictions_vs_truths(predictions, truths, series_names=None, 
                               title='True vs Predicted Values', 
                               xaxis_title='True Values', 
                               yaxis_title='Predicted Values', 
                               legend_title='Series',
                               show_diagonal=True, 
                               show_points_by_default=True,
                               filepath=None):
    """
    Plots true values vs predictions for multiple series using Plotly.
    
    Parameters:
    predictions (np.ndarray): 2D array of predicted values (rows: series, columns: points).
    truths (np.ndarray): 2D array of true values (same shape as predictions).
    series_names (list, optional): Names for each series. Defaults to generic names.
    title (str, optional): Plot title.
    xaxis_title (str, optional): X-axis title.
    yaxis_title (str, optional): Y-axis title.
    legend_title (str, optional): Legend title.
    show_diagonal (bool, optional): Whether to show the diagonal line.
    show_points (bool, optional): Whether points are visible by default (False sets to 'legendonly').
    path (str, optional): Path to save the plot as HTML.
    """
    # Validate input shapes
    if predictions.shape != truths.shape:
        raise ValueError("predictions and truths must have the same shape")
    
    n_series = predictions.shape[0]
    
    # Generate default series names if not provided
    if series_names is None:
        series_names = [f'Series {i+1}' for i in range(n_series)]
    elif len(series_names) != n_series:
        raise ValueError("series_names length must match number of series")
    
    # Create traces for each series
    traces = []
    for i in range(n_series):
        trace = go.Scatter(
            x=truths[i],
            y=predictions[i],
            mode='markers',
            name=series_names[i],
            visible=None if show_points_by_default else 'legendonly'
        )
        traces.append(trace)
    
    # Create diagonal line trace if enabled
    if show_diagonal:
        all_values = np.concatenate([truths.flatten(), predictions.flatten()])
        min_val, max_val = min(all_values), max(all_values)
        diagonal_trace = go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name='Perfect Prediction'
        )
        traces.append(diagonal_trace)
    
    # Create figure and update layout
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title=legend_title
    )
    
    # Save to HTML if path is provided
    if filepath:
        fig.write_html(filepath)
    
    return fig



# ! DEPRECATED (TOO OVERSIZED HTML FILE)
# def plot_dataset_results(
#         train_data, val_data,
#         colors, title,
#         train_labels, val_labels,
#         train_means, val_means,
#         train_stds, val_stds,
#         maxaes, maxaes_labels,
#         filepath
# ):
    
#     # Three figures. One for the matrix elements (HUGE), one for stats such as mean and std, and one for max abs difference (maxae) because it needs a very long fig. 

#     fig = go.Figure()
#     n_train_samples = len(train_data[0])
#     n_val_samples = len(val_data[0])

#     train_data = [[train_data[k][i].data for i in range(n_train_samples)] for k in range(len(train_data))]
#     val_data = [[val_data[k][i].data for i in range(n_val_samples)] for k in range(len(val_data))]
    

#     # ====== TRAINING DATA ======
#     matrix_traces = []
#     mean_traces = []
#     std_traces = []
#     for i in range(n_train_samples):
#         # Training matrix elements
#         trace = go.Scatter(
#             x=train_data[0][i],
#             y=train_data[1][i],
#             mode='markers',
#             marker=dict(
#                 # symbol='dash',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=0)
#             ),
#             name=f'Training sample {i}',
#             text=train_labels[i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='training',
#             # legendgrouptitle="Training samples",
#             showlegend=True
#         )
#         matrix_traces.append(trace)

#         # Training means
#         trace = go.Scatter(
#             x=[train_means[0][i]],
#             y=[train_means[1][i]],
#             mode='markers',
#             marker=dict(
#                 symbol='square',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=0)
#             ),
#             name=f'Mean training {i}',
#             text=maxaes_labels[0][i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='training_mean',
#             visible=False,
#             showlegend=True,
#         )
#         mean_traces.append(trace)

#         # Training std
#         trace = go.Scatter(
#             x=[train_stds[0][i]],
#             y=[train_stds[1][i]],
#             mode='markers',
#             marker=dict(
#                 symbol='triangle-up',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=0)
#             ),
#             name=f'Std training {i}',
#             text=maxaes_labels[0][i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='training_std',
#             visible=False,
#             showlegend=True,
#         )
#         std_traces.append(trace)

#     # === Validation ===
#     for i in range(n_val_samples):
#         trace = go.Scatter(
#             x=val_data[0][i],
#             y=val_data[1][i],
#             mode='markers',
#             marker=dict(
#                 symbol='circle-open',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=1,)
#             ),
#             name=f'Validation sample {i}',
#             text=val_labels[i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='validation',
#             # legendgrouptitle="Validation samples",
#             showlegend=True
#         )
#         matrix_traces.append(trace)

#         # Val means
#         trace = go.Scatter(
#             x=[val_means[0][i]],
#             y=[val_means[1][i]],
#             mode='markers',
#             marker=dict(
#                 symbol='square-open',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=1,)
#             ),
#             name=f'Mean val {i}',
#             text=maxaes_labels[1][i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='val_mean',
#             visible=False,
#             showlegend=True,
#         )
#         mean_traces.append(trace)

#         # Validation std
#         trace = go.Scatter(
#             x=[val_stds[0][i]],
#             y=[val_stds[1][i]],
#             mode='markers',
#             marker=dict(
#                 symbol='triangle-up-open',
#                 size=5,
#                 color=colors[i % len(colors)],
#                 line=dict(width=0)
#             ),
#             name=f'Std validation {i}',
#             text=maxaes_labels[1][i],
#             # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
#             legendgroup='val_std',
#             visible=False,
#             showlegend=True,
#         )
#         std_traces.append(trace)
        


#     # Add identity line
#     train_flattened_data = ([np.min(train_data[0][i]) for i in range(n_train_samples)], [np.max(train_data[0][i]) for i in range(n_val_samples)]) # [train, val]
#     train_flattened_data = [train_flattened_data[k] for k in range(len(train_flattened_data))]
#     min, max = np.min(train_flattened_data), np.max(train_flattened_data)
#     diagonal_trace = go.Scatter(
#         x=[min, max],
#         y=[min, max],
#         mode='lines',
#         line=dict(color='black', dash='solid'),
#         name='Ideal'
#     )


#     # Last dropdown: Max Absolute error
#     error_trace_train = go.Scatter(
#         x=maxaes[0],
#         y=maxaes_labels[0],
#         mode='markers',
#         marker=dict(
#             symbol='x',
#             size=6,
#             color="blue"
#         ),
#         name='Training',
#         showlegend=False  # optional: hide legend for this simple plot
#     )
#     # Last dropdown: Max Absolute error
#     error_trace_val = go.Scatter(
#         x=maxaes[1],
#         y=maxaes_labels[1],
#         mode='markers',
#         marker=dict(
#             symbol='x',
#             size=6,
#             color="red"
#         ),
#         name='Validation',
#         showlegend=False  # optional: hide legend for this simple plot
#     )
#     # zero_line_trace = go.Scatter(
#     #     x=[0, 0],
#     #     y=[maxaes_labels[1][-1], maxaes_labels[0][-1]],  # or your preferred Y range
#     #     mode='lines',
#     #     line=dict(color='black', dash='dash'),
#     #     name='zero',
#     #     showlegend=False
#     # )






#     # Create figure and update layout
#     traces = matrix_traces + mean_traces + std_traces + [error_trace_train] + [error_trace_val] + [diagonal_trace]
#     fig = go.Figure(data=traces)
#     fig.update_layout(
#         width=1000,
#         height=1000,
#         title=title,
#         # xaxis_title='True Values',
#         # yaxis_title='Predicted Values',
#         legend_title='Legend',
#         # hovermode='closest',
#         # template='plotly_white',
#         xaxis=dict(
#             title='True Values',
#             tickformat=".2f"
#         ),
#         yaxis=dict(
#             title='Predicted Values',
#             tickformat=".2f"
#         )
#     )

#     # Add dropdown
#     data_true = np.concatenate(train_data[0] + val_data[0], axis=0)
#     data_pred = np.concatenate(train_data[1] + val_data[1], axis=0)
#     min_x_data = data_true.min()
#     max_x_data = data_true.max()
#     min_y_data = data_pred.min()
#     max_y_data = data_pred.max()

#     min_x_mean = np.min([train_means[0].min(), val_means[0].min()])
#     max_x_mean = np.max([train_means[0].max(), val_means[0].max()])
#     min_y_mean = np.min([train_means[1].min(), val_means[1].min()])
#     max_y_mean = np.max([train_means[1].max(), val_means[1].max()])

#     min_x_std = np.min([train_stds[0].min(), val_stds[0].min()])
#     max_x_std = np.max([train_stds[0].max(), val_stds[0].max()])
#     min_y_std = np.min([train_stds[1].min(), val_stds[1].min()])
#     max_y_std = np.max([train_stds[1].max(), val_stds[1].max()])

#     fig.update_layout(
#         updatemenus=[
#             dict(
#                 buttons=[
#                     dict(
#                         label="SISL Hamiltonian elements",
#                         method="update",
#                         args=[{"visible": [True]*len(matrix_traces) + [False]*len(mean_traces) + [False]*len(std_traces) + [False]*2 + [True]},
#                             {
#                                 "xaxis": {"range": [min_x_data-0.05*min_x_data, max_x_data+0.05*max_x_data]},
#                                 "yaxis": {"range": [min_y_data-0.05*min_y_data, max_y_data+0.05*max_y_data]},
#                             }]
#                     ),
#                     dict(
#                         label="Mean",
#                         method="update",
#                         args=[
#                             {"visible": [False]*len(matrix_traces) + [True]*len(mean_traces) + [False]*len(std_traces) + [False]*2 + [True]},
#                             {
#                                 "xaxis": {"range": [min_x_mean-0.0005*min_x_mean, max_x_mean+0.0005*max_x_mean]},
#                                 "yaxis": {"range": [min_y_mean-0.0005*min_y_mean, max_y_mean+0.0005*max_y_mean]},
#                             }

#                         ]
#                     ),
#                     dict(
#                         label="Std",
#                         method="update",
#                         args=[
#                             {"visible": [False]*len(matrix_traces) + [False]*len(mean_traces) + [True]*len(std_traces) + [False]*2 + [True]},
#                             {
#                                 "xaxis": {"range": [min_x_std-0.0005*min_x_std, max_x_std+0.0005*max_x_std]},
#                                 "yaxis": {"range": [min_y_std-0.0005*min_y_std, max_y_std+0.0005*max_y_std]},
#                             }

#                         ]
#                     ),
#                     dict(
#                         label="Max Absolute Error",
#                         method="update",
#                         args=[
#                             {"visible": [False]*len(matrix_traces) + [False]*len(mean_traces) + [False]*len(std_traces) + [True]*2 + [False]},
#                             {"xaxis": {"title": "Max Absolute Error"},
#                             "yaxis": {
#                                 "title": "Structures",
#                                 "type": "category",
#                                 #  "categoryorder": "array",
#                                 "categoryarray": maxaes_labels,
#                                 "autorange": "reversed"
#                             },
#                             "showlegend": [False]*len(matrix_traces + mean_traces + std_traces) + [False]*2 + [False]}
#                         ]
#                     )


#                 ],
#                 direction="down",
#                 pad={"r": 10, "t": 10},
#                 showactive=True,
#                 x=0.3,
#                 xanchor="left",
#                 y=1.1,
#                 yanchor="top"
#             ),
#         ]
#     )

#     # Save to HTML if path is provided
#     if filepath:
#         f = open(filepath, "w")
#         f.close()
#         with open(filepath, 'a') as f:
#             f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
#         f.close()
        
#         with open(f"{str(filepath)[:-4]}.json", "w") as f:
#             f.write(fig.to_json())

    
#     return fig