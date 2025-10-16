from typing import Any, Tuple, Dict, Type, Callable, Union

# Import all loss functions from graph2mat
from graph2mat.core.data.metrics import (
    OrbitalMatrixMetric,
    block_type_mse,
    block_type_mae,
    block_type_mape,
    block_type_mapemaemix,
    block_type_mapemsemix,
    block_type_mapestdmix,
    elementwise_mse,
    node_mse,
    edge_mse,
    block_type_mse_threshold,
    block_type_mse_sigmoid_thresh,
    block_type_mae_sigmoid_thresh,
    normalized_density_error,
)



def _isnan(values):
    """NaN checking compatible with both torch and numpy"""
    return values != values

def _remove_zeros_simultaneously(values_1, values_2):
    """Remove zero values from a tensor or array values_1, and removes the corresponding entries in values_2."""
    nonzero = values_1 != 0
    return values_1[nonzero], values_2[nonzero]

def get_predictions_error_nonzero(
    nodes_pred, nodes_ref, edges_pred, edges_ref, remove_nan=True,
):
    """Returns errors for both nodes and edges, removing NaN values and zero values."""
    nodes_ref, nodes_pred = _remove_zeros_simultaneously(nodes_ref, nodes_pred)
    edges_ref, edges_pred = _remove_zeros_simultaneously(edges_ref, edges_pred)

    node_error = nodes_pred - nodes_ref

    if remove_nan:
        notnan = ~_isnan(edges_ref)

        edge_error = edges_ref[notnan] - edges_pred[notnan]
    else:
        edge_error = edges_ref - edges_pred

    return node_error, edge_error



@OrbitalMatrixMetric.from_metric_func
def block_type_mse_nonzero_globalsquarenorm(
    nodes_pred, nodes_ref, edges_pred, edges_ref, log_verbose=False, **kwargs
) -> Tuple[float, Dict[str, float]]:
    node_error, edge_error = get_predictions_error_nonzero(
        nodes_pred, nodes_ref, edges_pred, edges_ref
    )
        
    norm = 1.0 if kwargs["norm"] is None else float(kwargs["norm"])
    print("Using global normalization factor: ", norm)

    node_loss = (node_error**2).mean() * 1/(norm**2)
    edge_loss = (edge_error**2).mean() * 1/(norm**2)

    stats = {
        "node_rmse": node_loss ** (1 / 2),
        "edge_rmse": edge_loss ** (1 / 2),
    }

    if log_verbose:
        abs_node_error = abs(node_error)
        abs_edge_error = abs(edge_error)

        stats.update(
            {
                "node_mean": abs_node_error.mean(),
                "edge_mean": abs_edge_error.mean(),
                "node_std": abs_node_error.std(),
                "edge_std": abs_edge_error.std(),
                "node_max": abs_node_error.max(),
                "edge_max": abs_edge_error.max(),
            }
        )

    return node_loss + edge_loss, stats