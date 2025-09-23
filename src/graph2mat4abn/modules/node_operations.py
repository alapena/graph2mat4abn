import torch
from torch import nn
from e3nn import o3

class HamGNNInspiredNodeBlock(nn.Module):
    """
    A more expressive, HamGNN-style replacement for f_n.

    Parameters
    ----------
    irreps_in : o3.Irreps | str | list[o3.Irreps | str]
        Irreducible representations of each input tensor (one per
        keyword that will be passed at forward time).

    irreps_out : o3.Irreps | str
        Target irreps shape – must match what Graph2Mat expects for
        the on-site block of this atom type.

    Notes
    -----
    * Preserves full SE(3)/O(3) equivariance (all operations are built
      from e3nn objects).
    * For every input tensor `x` it computes
          y = TP(x, x) + Skip(x)
      where
          TP   : FullyConnectedTensorProduct (bilinear, learnable)
          Skip : Linear mixing (learnable)
      and then sums the contributions from all keyword tensors.
    * Output shape: [n_nodes, irreps_out.dim] — identical to the old
      TensorSquare block.
    """
    def __init__(self, irreps_in, irreps_out):
        super().__init__()

        # allow a single Irreps or list-like
        if isinstance(irreps_in, (o3.Irreps, str)):
            irreps_in = [irreps_in]

        self.blocks = nn.ModuleList()
        self.irreps_out = o3.Irreps(irreps_out)

        for this_irrep in irreps_in:
            ir_in = o3.Irreps(this_irrep)

            # (1) Bilinear self-coupling
            tp = o3.FullyConnectedTensorProduct(
                ir_in, ir_in, self.irreps_out,
                shared_weights=True, internal_weights=True
            )

            # (2) Residual skip (linear in channels, equivariant)
            skip = o3.Linear(ir_in, self.irreps_out,
                             internal_weights=True, shared_weights=True)

            self.blocks.append(nn.ModuleDict({
                "tp": tp,
                "skip": skip
            }))

    # ---------------------------------------------------------------------
    def forward(self, **node_kwargs):
        """
        Parameters
        ----------
        node_kwargs : Dict[str, Tensor]
           Each value is a tensor of shape [n_nodes, irreps_in[i].dim].

        Returns
        -------
        Tensor
            Shape [n_nodes, irreps_out.dim]
        """
        assert len(node_kwargs) == len(self.blocks), (
            f"Expected {len(self.blocks)} tensors, "
            f"got {len(node_kwargs)} instead."
        )

        out = None
        for (tensor, mods) in zip(node_kwargs.values(), self.blocks):
            y = mods["tp"](tensor, tensor) + mods["skip"](tensor)
            out = y if out is None else out + y  # accumulate

        return out
