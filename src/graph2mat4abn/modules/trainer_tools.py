import torch

def force_zeroes_below_threshold(data, model_predictions, threshold, update_true_labels=False):
    """Sets model predictions to zero where true values are under a certain threshold.

    Args:
        data (TorchBasisMatrixData or TorchBasisMatrixDataBatch): Graph2Mat data object.
        model_predictions (Dict): Dictionary containing "edge_labels" and "node_labels" as the predictions of the model.
        threshold (float): Minimum energy below to which force zero.
    """
    # Edges
    mask = torch.abs(data.edge_labels) < threshold
    new_labels = torch.where(mask, 0, model_predictions["edge_labels"])
    model_predictions["edge_labels"] = new_labels.to(data.edge_labels.device)

    if update_true_labels:
        data.edge_labels[mask] = 0


    # Nodes
    mask = torch.abs(data.point_labels) < threshold
    new_labels = torch.where(mask, 0, model_predictions["node_labels"])
    model_predictions["node_labels"] = new_labels.to(data.point_labels.device)

    if update_true_labels:
        data.point_labels[mask] = 0