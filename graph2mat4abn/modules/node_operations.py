import torch
from e3nn.nn import FullyConnectedNet
from e3nn import o3

# ! EXAMPLE:
class CustomOperation(torch.nn.Module):

    def __init__(self, irreps_in, irreps_out, config):
        super().__init__()
        print("INITIALIZING OPERATION")
        print("INPUT NODE FEATS IRREPS:", irreps_in)
        print("IRREPS_OUT:", irreps_out)
        print("")

        self.linear = o3.Linear(irreps_in, irreps_in)

    def __call__(self, node_feats):
        # print(node_feats)

        # This return will create an error. Instead, you should
        # produce something of irreps_out.
        a
        return node_feats
    

# ! DEPRECATED:
class UniversalSetApproximator(torch.nn.Module):

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, config: dict):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.config = config

        # self.hidden_neurons = config.get("hidden_neurons", [64, 64])
        # activation = config.get("activation", "silu")
        self.hidden_neurons = [64, 64]
        self.activation = torch.nn.ReLU()
        
        # Step 1 (MLP_φ in the formula)
        self.mlp1 = FullyConnectedNet(
            [irreps_in.num_irreps] + self.hidden_neurons + [irreps_in.num_irreps],
            act=self.activation
        )
        
        # Step 2 (MLP_θ in the formula)
        self.mlp2 = FullyConnectedNet(
            [irreps_in.num_irreps] + self.hidden_neurons + [irreps_out.num_irreps],
            act=self.activation
        )
        
        # # Equivariant linear layer
        # self.output_linear = o3.Linear(
        #     irreps_out,
        #     irreps_out,
        #     internal_weights=True,
        #     shared_weights=True
        # )

    def __call__(self, node_feats):
        print("node_feats.shape: ", node_feats.shape)

        
        
        return node_feats
    

