import plotly.graph_objects as go
import plotly.colors as pc
from plotly.colors import sample_colorscale
import numpy as np
from collections import defaultdict
from plotly.subplots import make_subplots

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



def plot_diagonal(
    true_values, pred_values, *group_keys, # 1D array of elements.
    title="Default title", title_x="X axis", title_y="Y axis", colors=None, filepath=None,
    group_by="shift", legendtitle=None,
):

    fig = go.Figure()
    
    # ====== TRAINING DATA ======
    traces = []
    # # Training matrix elements
    # trace = go.Scattergl(
    #     x=true_values,
    #     y=pred_values,
    #     mode='markers',
    #     marker=dict(
    #         # symbol='dash',
    #         size=5,
    #         # color=colors[i % len(colors)],
    #         line=dict(width=0)
    #     ),
    #     name=f'Matrix elements',
    #     text=labels,
    #     # hovertemplate='True: %{x:.2f}<br>Pred: %{y:.2f}<br>%{text}',
    #     # legendgroup='training',
    #     # legendgrouptitle="Training samples",
    #     # showlegend=True
    # )

    # Plot matrix elements but split on shifts
    if group_by == "shift":

        # Group by shift
        true_grouped = defaultdict(list)
        pred_grouped = defaultdict(list)
        labels_grouped = defaultdict(list)
        for orb_in, orb_out, isc, atom_in, atom_out, true_h, pred_h in zip(*group_keys, true_values, pred_values):
            true_grouped[isc].append(true_h)
            pred_grouped[isc].append(pred_h)
            labels_grouped[isc].append(f"{str(orb_in)} -> {str(orb_out)} {str(isc)} {str(atom_in)} -> {str(atom_out)}")

        # Plot grouped by shifts
        for i, isc in enumerate(pred_grouped.keys()):
            
            trace = go.Scattergl(
                x=true_grouped[isc],
                y=pred_grouped[isc],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=5,
                    line=dict(width=0),
                ),
                name=isc,
                text=labels_grouped[isc],
            )
            traces.append(trace)


    elif group_by == "orbs":

        # Group by orbs
        true_grouped = defaultdict(list)
        pred_grouped = defaultdict(list)
        name_grouped = defaultdict(list)
        labels_grouped = defaultdict(list)
        for orb_in, orb_out, isc, atom_in, atom_out, true_h, pred_h in zip(*group_keys, true_values, pred_values):
            key = (orb_in, orb_out, atom_in, atom_out)
            true_grouped[key].append(true_h)
            pred_grouped[key].append(pred_h)
            label = f"{str(orb_in)} -> {str(orb_out)}, atm{str(atom_in)} to atm{str(atom_out)}"
            if label not in name_grouped[key]:
                name_grouped[key].append(label)
            labels_grouped[key].append(f"{str(orb_in)} -> {str(orb_out)} {str(isc)} {str(atom_in)} -> {str(atom_out)}")
        
        # Plot grouped by orbs
        for i, isc in enumerate(pred_grouped.keys()):
            trace = go.Scattergl(
                x=true_grouped[isc],
                y=pred_grouped[isc],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    size=5,
                    line=dict(width=0),
                    # color='blue'
                ),
                name=name_grouped[isc][0],
                text=labels_grouped[isc],
                visible='legendonly',
            )
            traces.append(trace)

    # Add identity line
    all_data = np.concatenate([true_values, pred_values])
    vmin, vmax = np.min(all_data), np.max(all_data)
    diagonal_trace = go.Scatter(
        x=[vmin, vmax],
        y=[vmin, vmax],
        mode='lines',
        line=dict(color='black', dash='dash'),
        name='Ideal',
    )
    traces.append(diagonal_trace)


    # Create figure and update layout
    fig = go.Figure(data=traces)
    fig.update_layout(
        width=900,
        height=900,
        title=title,
        xaxis=dict(
            title=title_x,
            tickformat=".2f"
        ),
        yaxis=dict(
            title=title_y,
            tickformat=".2f"
        ),
        legend=dict(title=dict(text=legendtitle))
    )

    # Save to HTML if path is provided
    if filepath:
        f = open(filepath, "w")
        f.close()
        with open(filepath, 'a') as f:
            f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        f.close()
        
        # with open(f"{str(filepath)[:-4]}.json", "w") as f:
        #     f.write(fig.to_json())

    else:
        fig.show()

    return fig



def plot_diagonal_rows(predictions, truths, series_names=None, 
                               x_error_perc = None,
                               y_error_perc = None,
                               title='True vs Predicted Values', 
                               xaxis_title='True Values', 
                               yaxis_title='Predicted Values', 
                               legend_title='Series',
                               show_diagonal=True, 
                               show_points_by_default=True,
                               showlegend=True,
                               filepath=None):

    # Validate input shapes
    if predictions.shape != truths.shape:
        raise ValueError("predictions and truths must have the same shape")
    
    n_series = predictions.shape[0]
    colors = sample_colorscale('Bluered', [i/(n_series-1) for i in range(n_series)][::-1])
    
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
            marker=dict(color=colors[i]),
            name=series_names[i],
            visible=None if show_points_by_default else 'legendonly',
            error_x=dict(
                type='percent',
                value=x_error_perc,
                visible=True
            ) if x_error_perc is not None else None,
            error_y=dict(
                type='percent',
                value=y_error_perc,
                visible=True
            ) if y_error_perc is not None else None,
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
        legend_title=legend_title,
        showlegend=showlegend
    )
    
    # Save to HTML if path is provided
    if filepath:
        fig.write_html(filepath)
    
    return fig



def plot_energy_bands(k_path, array_true, array_pred=None, titles_true=None, titles_pred=None, xlabel=None, ylabel=None, title=None, filepath=None):

    if array_pred.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    num_rows = array_pred.shape[0]
    num_cols = array_pred.shape[1]

    # Default color sequence (you can customize this)

    if titles_pred is None:
        titles_pred = [f'Pred Row {i+1}' for i in range(num_rows)]
    elif len(titles_pred) != num_rows:
        raise ValueError("Number of titles_pred must match number of rows")

    if titles_true is None:
        titles_true = [f'True Row {i+1}' for i in range(num_rows)]
    elif len(titles_true) != num_rows:
        raise ValueError("Number of titles_true must match number of rows")



    fig = go.Figure()

    colors = pc.qualitative.Plotly  # or another palette
    num_colors = len(colors)

    for row in range(num_rows):
        color = colors[row % num_colors]

        # Add predicted values as dashed line 
        fig.add_trace(go.Scatter(
            x=k_path,
            y=array_pred[row],
            mode='lines',
            line=dict(dash='dash', color=color),
            name=titles_pred[row],
            # opacity=0.8,
            legendgroup=f'group{row}',
            showlegend=True
        ))

        if array_true is not None:
            # Add true values as solid line with same color 
            fig.add_trace(go.Scatter(
                x=k_path,
                y=array_true[row],
                mode='lines',
                line=dict(dash='solid', color=color),
                name=titles_true[row],
                # opacity=0.8,
                legendgroup=f'group{row}',
                showlegend=True
            ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        # hovermode='x unified',
        height=800,
        xaxis_title_standoff=15,
    )

    fig.update_xaxes(
        showticklabels=False,
    )

    # === Output ===
    if filepath:
        filepath = filepath
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath))
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
    else:
        fig.show()

    return fig



def plot_bands(k_len, bands, k_idx, k_label, predicted_bands=None, filepath=None):
    """
    Plot band structure using Plotly.

    Parameters:
    - k_len: array-like, k-point distances
    - bands: 2D array, shape (n_kpoints, n_bands), true bands
    - k_idx: list of indices where vertical lines are drawn
    - k_label: list of labels for xticks
    - predicted_bands: optional 2D array, same shape as `bands`
    """
    fig = go.Figure()
    num_bands = bands.shape[1]

    # True bands: solid black
    for i in range(num_bands):
        fig.add_trace(go.Scatter(
            x=k_len,
            y=bands[:, i],
            mode='lines',
            name=f'True Band {i+1}',
            line=dict(color='black', width=1, dash='solid'),
            legendgroup=f'Band {i+1}',
            showlegend=True if predicted_bands is None else False
        ))

    # Predicted bands: dashed black
    if predicted_bands is not None:
        for i in range(num_bands):
            fig.add_trace(go.Scatter(
                x=k_len,
                y=predicted_bands[:, i],
                mode='lines',
                name=f'Predicted Band {i+1}',
                line=dict(color='black', width=1, dash='dash'),
                legendgroup=f'Band {i+1}',
                showlegend=True
            ))

    # Vertical lines
    for idx in k_idx:
        fig.add_shape(type="line",
                      x0=k_len[idx], y0=bands.min(), x1=k_len[idx], y1=bands.max(),
                      line=dict(color="black", width=1))

    # Layout
    fig.update_layout(
        xaxis=dict(
            title="k (1/nm)",
            tickmode='array',
            tickvals=[k_len[i] for i in k_idx],
            ticktext=k_label,
            ticks='',  # Hide tick marks
            showticklabels=True,
            range=[0, k_len.max()]
        ),
        yaxis=dict(title="Energy (eV)"),
        margin=dict(l=50, r=20, t=20, b=50),
        showlegend=True
    )

    # === Output ===
    if filepath is not None:
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath), height=1200, width=900,)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
        
    else:
        fig.show()

    return fig



def plot_dos(energies, dos, predicted_dos=None, filepath=None):
    """
    Plot Density of States (DOS) vertically with Energy on y-axis.

    Parameters:
    - energies: array-like, energy values (eV)
    - dos: array-like, true DOS values
    - predicted_dos: optional array-like, predicted DOS values (same shape as dos)
    """
    fig = go.Figure()

    # True DOS
    fig.add_trace(go.Scatter(
        x=dos,
        y=energies,
        mode='lines',
        name='True',
        line=dict(color='black', width=1, dash='solid')
    ))

    # Predicted DOS
    if predicted_dos is not None:
        fig.add_trace(go.Scatter(
            x=predicted_dos,
            y=energies,
            mode='lines',
            name='Pred',
            line=dict(color='black', width=1, dash='dash')
        ))

    # Layout
    fig.update_layout(
        xaxis=dict(title='DOS (1/eV)'),
        yaxis=dict(title='Energy (eV)'),
        margin=dict(l=50, r=50, t=20, b=20),
        showlegend=True
    )

    # === Output ===
    if filepath is not None:
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath), height=1200, width=900,)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
        
    else:
        fig.show()

    return fig


def combine_band_and_dos(fig_band, fig_dos, filepath=None):
    """
    Combine band structure and DOS plots side by side into a single figure.

    Parameters:
    - fig_band: Plotly figure from plot_bands()
    - fig_dos: Plotly figure from plot_dos()

    Returns:
    - Combined Plotly figure
    """
    # Create 1-row, 2-column subplot
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        column_widths=[0.75, 0.25],
                        horizontal_spacing=0.02,
                        specs=[[{"type": "xy"}, {"type": "xy"}]])

    # Add band traces to subplot (1,1)
    for trace in fig_band.data:
        fig.add_trace(trace, row=1, col=1)

    # Add DOS traces to subplot (1,2)
    for trace in fig_dos.data:
        fig.add_trace(trace, row=1, col=2)

    # Update layout
    fig.update_layout(
        xaxis=dict(title='k (1/nm)'),  # subplot (1,1)
        xaxis2=dict(title='DOS (1/eV)'),  # subplot (1,2)
        yaxis=dict(title='Energy (eV)'),  # shared y-axis
        showlegend=True,
        margin=dict(l=50, r=20, t=20, b=40)
    )

    # === Output ===
    if filepath is not None:
        if filepath.suffix.lower() == ".html":
            fig.write_html(str(filepath))
        elif filepath.suffix.lower() == ".png":
            fig.write_image(str(filepath), height=1200, width=900,)
        else:
            raise ValueError(f"Unsupported file extension: {filepath.suffix}")
        
    else:
        fig.show()

    return fig