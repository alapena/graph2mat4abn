import torch
import e3nn as o3

from mace.modules import RadialEmbeddingBlock, EquivariantProductBasisBlock
from mace.modules.utils import get_edge_vectors_and_lengths

from graph2mat4abn.tools import z_one_hot
from graph2mat4abn.tools import get_object_from_module


class EmbeddingBase(torch.nn.Module):
    def __init__(self, config, orbitals):
        super(EmbeddingBase, self).__init__()

        embeddings_config = config["model"]["embedding"]
        self.device = "cpu" #config["device"]

        # Define the irreducible representations for the node attributes and features.
        node_attr_irreps = o3.Irreps([(embeddings_config["num_elements"], (0, 1))]) # E.g. [(10, (0,1))]
        hidden_irreps = o3.Irreps(embeddings_config["hidden_irreps"]) # E.g. "8x0e+8x1o"
        node_feats_irreps = o3.Irreps([(hidden_irreps.count(o3.Irrep(0, 1)), (0, 1))]) # Counting how many Irrep(0, 1) there are inside hidden_irreps.

        # Linear transformation from node attributes to node features.
        # / I think this is the same as torch.nn.Linear
        self.node_embedding = o3.Linear(
            node_attr_irreps,
            node_feats_irreps,
            shared_weights=True,
            internal_weights=True,
        )

        # Radial embedding block using Bessel functions and polynomial cutoffs.
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=embeddings_config["r_max"],
            num_bessel=embeddings_config["num_bessel"],
            num_polynomial_cutoff=embeddings_config["num_polynomial_cutoff"],
            radial_type=embeddings_config["radial_type"],
            distance_transform=embeddings_config["distance_transform"],
        )

        # Angular embedding using spherical harmonics.
        sh_irreps = o3.Irreps.spherical_harmonics(embeddings_config["max_ell"])
        self.angular_embedding = o3.SphericalHarmonics(sh_irreps, normalize=True, normalization="component")

        # Element encoding configuration
        self.orbitals = orbitals
        self.nr_bit = embeddings_config["nr_bits"]

    def forward(self, data):

        # Encode atomic numbers into binary orbital-based representation. 
        atom_types = data.metadata['atom_types'].unsqueeze(1) # The reshape is for format reasons
        one_hot_z = z_one_hot(atom_types, orbitals=self.orbitals, nr_bits=self.nr_bit).to(self.device)

        # Input node descriptors.
        node_feats = one_hot_z

        # Calculate edge vectors and their lengths (distances).
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data.positions,
            edge_index=data.edge_index,
            shifts=data.shifts,
        )

        # Apply node embedding.
        node_feats = self.node_embedding(node_feats)
        

        # Apply radial and angular embeddings for edges.
        radial_embedding = self.radial_embedding(
            lengths,
            node_feats,
            data.edge_index,
            atom_types
        )
        angular_embedding = self.angular_embedding(vectors)

        # Bundle the embeddings.
        embedding_collection = {
            "nodes": {
                "one_hot": one_hot_z,
                "node_features": node_feats,
            },
            "edges": {
                "radial_embedding": radial_embedding,
                "angular_embedding": angular_embedding,
            }
        }

        return embedding_collection
       
    

class MACEDescriptor(torch.nn.Module):
    def __init__(self, atomic_descriptors_config):
        super(MACEDescriptor, self).__init__()

        # --- Irreps definitions ---
        node_attr_irreps = o3.Irreps([(atomic_descriptors_config["num_elements"], (0, 1))])  # One-hot per element (scalar-even)

        # Extract number of scalar-even irreps from hidden_irreps
        hidden_irreps = o3.Irreps(atomic_descriptors_config["hidden_irreps"])
        num_scalar_irreps = hidden_irreps.count(o3.Irrep(0, 1))
        node_feats_irreps = o3.Irreps([(num_scalar_irreps, (0, 1))])

        sh_irreps = o3.Irreps.spherical_harmonics(atomic_descriptors_config["max_ell"])  # Angular features

        radial_out_dim = atomic_descriptors_config["radial_embedding.out_dim"]
        edge_feats_irreps = o3.Irreps(f"{radial_out_dim}x0e")  # Radial embeddings as scalar-even

        hidden_irreps_out = hidden_irreps  # Output IRs remain same

        # Determine output irreps of interaction (spherical harmonics ⊗ scalar features)
        interaction_irreps = (sh_irreps * num_scalar_irreps).sort()[0].simplify()

        # Support for correlation order per layer
        if isinstance(atomic_descriptors_config["correlation"], int):
            correlation = [atomic_descriptors_config["correlation"]] * atomic_descriptors_config["num_interactions"]

        # --- First Interaction Layer ---
        interaction_cls_first = get_object_from_module(atomic_descriptors_config["interaction_cls_first"], "mace.modules")
        first_interaction = interaction_cls_first(
            node_attrs_irreps=node_attr_irreps,
            node_feats_irreps=node_feats_irreps,
            edge_attrs_irreps=sh_irreps,
            edge_feats_irreps=edge_feats_irreps,
            target_irreps=interaction_irreps,
            hidden_irreps=hidden_irreps,
            avg_num_neighbors=atomic_descriptors_config["avg_num_neighbors"],
            radial_MLP=atomic_descriptors_config["radial_mlp"],
            cueq_config=None,
        )

        self.interactions = torch.nn.ModuleList([first_interaction])

        # Determine whether to use self-connection (important for residual-based models)
        use_sc_first = "Residual" in str(atomic_descriptors_config["interaction_cls_first"])

        first_product = EquivariantProductBasisBlock(
            node_feats_irreps=first_interaction.target_irreps,
            target_irreps=hidden_irreps,
            correlation=correlation[0],
            num_elements=atomic_descriptors_config["num_elements"],
            use_sc=use_sc_first,
            cueq_config=None,
        )

        self.products = torch.nn.ModuleList([first_product])

        # --- Remaining Interaction-Product Blocks ---
        for i in range(atomic_descriptors_config["num_interactions"] - 1):
            interaction_cls = get_object_from_module(atomic_descriptors_config["interaction_cls"], "mace.modules")
            interaction = interaction_cls(
                node_attrs_irreps=node_attr_irreps,
                node_feats_irreps=hidden_irreps_out,
                edge_attrs_irreps=sh_irreps,
                edge_feats_irreps=edge_feats_irreps,
                target_irreps=interaction_irreps,
                hidden_irreps=hidden_irreps_out,
                avg_num_neighbors=atomic_descriptors_config["avg_num_neighbors"],
                radial_MLP=atomic_descriptors_config["radial_mlp"],
                cueq_config=None,
            )

            product = EquivariantProductBasisBlock(
                node_feats_irreps=interaction_irreps,
                target_irreps=hidden_irreps_out,
                correlation=correlation[i + 1],
                num_elements=atomic_descriptors_config["num_elements"],
                use_sc=True,
                cueq_config=None,
            )

            self.interactions.append(interaction)
            self.products.append(product)

    def forward(self, embeddings, edge_index):
        """_summary_

        Args:
            data (_type_): Already preprocessed data.

        Returns:
            _type_: _description_
        """

        node_feats = embeddings["nodes"]["node_features"]

        node_feats_list = []
        for interaction, product in zip(self.interactions, self.products):
            node_feats, sc = interaction(
                node_attrs=embeddings["nodes"]["one_hot"],
                node_feats=node_feats,
                edge_attrs=embeddings["edges"]["angular_embedding"],
                edge_feats=embeddings["edges"]["radial_embedding"],
                edge_index=edge_index,
            )

            node_feats = product(
                node_feats=node_feats,
                sc=sc,
                node_attrs=embeddings["nodes"]["one_hot"],
            )

            node_feats_list.append(node_feats)

        # Concatenate features from all interaction layers
        node_feats_out = torch.cat(node_feats_list, dim=-1)

        # Final descriptor
        descriptors = {
            "nodes": {
                "node_env": node_feats_out,
            }
        }

        return descriptors