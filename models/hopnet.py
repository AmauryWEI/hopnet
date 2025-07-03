# File:         hopnet.py
# Date:         2024/07/12
# Description:  Higher-Order topological Phyiscs-informed Network

from typing import Callable

import torch
from torch_scatter import scatter

from .mlp import mlp


def propagate_messages(
    msgs: torch.Tensor,
    neighborhood: torch.Tensor,
    reduce: str = "sum",
    matmul_values: bool = True,
) -> torch.Tensor:
    """Message-passing from rows to columns

    Parameters
    ----------
    msgs : torch.Tensor
        Messages (1 per row)
    neighborhood : torch.Tensor
        Neighborhood matrix (rows x columns)
    reduce : str, optional
        Message aggregation, by default "sum"
    matmul_values : bool, optional
        True to multiply messages by the neighborhood matrix values, False to use 1s

    Returns
    -------
    torch.Tensor
        Aggregated messages for each column
    """
    n = neighborhood.coalesce()
    source, target = n.indices()
    msgs = msgs.index_select(0, source)
    if matmul_values:
        msgs = n.values().view(-1, 1) * msgs
    msgs = scatter(
        src=msgs,
        index=target,
        dim=0,
        reduce=reduce,
        # Specify dim_size to handle cases where some cells have no incoming messages
        dim_size=neighborhood.shape[1],
    )
    return msgs


class HOPNet_Layer(torch.nn.Module):
    """Single Message-Passing layer for HOPNet"""

    def __init__(self, channels: list[int], activation_func: Callable, mlp_layers: int):
        super().__init__()
        assert len(channels) == 5  # x0, x1, x2, x3, x4
        self.__x2_channels = channels[2]

        # Step 1: Enhancement of face embeddings
        self.__mlp_processor_0to2 = mlp(channels[0], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_1to2 = mlp(channels[1], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_4to2 = mlp(channels[4], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_2 = mlp(4 * channels[2], channels[2], channels[2], activation_func, mlp_layers)

        # Steps 2 and 3 : Inter-objects collisions
        self.__mlp_processor_2to3 = mlp(channels[2], channels[3], channels[3], activation_func, mlp_layers)
        self.__mlp_processor_3 = mlp(3 * channels[2], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_2p = mlp(2 * channels[2], channels[2], channels[2], activation_func, mlp_layers)

        # Step 4: Momentum transfer from faces to objects
        self.__mlp_processor_2to4 = mlp(channels[2], channels[4], channels[4], activation_func, mlp_layers)
        self.__mlp_processor_4 = mlp(2 * channels[4], channels[4], channels[4], activation_func, mlp_layers)

        # Step 5: Intra-object node update
        self.__mlp_processor_0to4 = mlp(channels[0], channels[4], channels[4], activation_func, mlp_layers)
        self.__mlp_processor_4to0 = mlp(channels[4], channels[4], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_0 = mlp(2 * channels[0], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_4p = mlp(2 * channels[3], channels[3], channels[3], activation_func, mlp_layers)

    def forward(self, x: tuple) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        (h0, h1, h2, h3_minus, h3_plus, h4, b02, b04, b12, b23, b24) = x

        # STEP 0: Process collisions (if any)
        if h3_minus is not None:
            assert h3_plus is not None
            assert h2 is not None
            assert b23 is not None
            assert b12 is not None
            assert b24 is not None

            # Compute messages to build up faces (equation 8 in the main manuscript)
            m0to2 = self.__mlp_processor_0to2(h0)
            m1to2 = self.__mlp_processor_1to2(h1)
            m4to2 = self.__mlp_processor_4to2(h4)

            # Create new face embedding (equation 9 in the main manuscript)
            m0to2 = propagate_messages(m0to2, b02)
            m1to2 = propagate_messages(m1to2, b12)
            m4to2 = propagate_messages(m4to2, b24.T)
            h2p = self.__mlp_processor_2(torch.cat((h2, m0to2, m1to2, m4to2), dim=1))

            # Handle collisions
            # Select sources with incidence matrix values as -1, and targets as +1
            faces, collisions = b23.coalesce().indices()
            values = b23.values()
            negative_faces_idx = (values < 0).nonzero(as_tuple=True)[0]
            positive_faces_idx = (values > 0).nonzero(as_tuple=True)[0]
            # Select the indices of the sources and targets (to index them in x2)
            negative_faces = faces[negative_faces_idx]
            positive_faces = faces[positive_faces_idx]
            # Create different messages for positive and negative mesh faces (part 1 of equation 10 in the main manuscript)
            m2to3_positive = self.__mlp_processor_2to3(h2p.index_select(0, positive_faces))
            m2to3_negative = self.__mlp_processor_2to3(h2p.index_select(0, negative_faces))
            # 1st pass: regroup the messages into a single matrix
            m2to3_plus = torch.zeros(
                (h2p.index_select(0, faces).shape[0], 2 * self.__x2_channels)
            ).to(h2p.device)
            m2to3_plus[positive_faces, self.__x2_channels :] = m2to3_positive
            m2to3_plus[negative_faces, : self.__x2_channels] = m2to3_negative
            # 2nd pass: regroup the messaged into another matrix
            m2to3_minus = torch.zeros_like(m2to3_plus)
            m2to3_minus[positive_faces, : self.__x2_channels] = m2to3_positive
            m2to3_minus[negative_faces, self.__x2_channels :] = m2to3_negative
            # Aggregate positive and negative messages to the collision embeddings
            m2to3_plus = propagate_messages(
                m2to3_plus, b23, reduce="sum", matmul_values=False
            )
            m2to3_minus = propagate_messages(
                m2to3_minus, b23, reduce="sum", matmul_values=False
            )
            # Compute new collision embeddings (part 2 of equation 10 in the main manuscript)
            h3p_plus = self.__mlp_processor_3(torch.cat((h3_plus, m2to3_plus), dim=1))
            h3p_minus = self.__mlp_processor_3(torch.cat((h3_minus, m2to3_minus), dim=1))

            # Distribute the x3_final matrices back to the faces
            m3to2_plus = scatter(
                src=h3p_plus[collisions[positive_faces_idx]],
                index=positive_faces,
                dim=0,
                reduce="sum",
                dim_size=h2p.shape[0],
            )
            m3to2_minus = scatter(
                src=h3p_minus[collisions[negative_faces_idx]],
                index=negative_faces,
                dim=0,
                reduce="sum",
                dim_size=h2p.shape[0],
            )
            m3to2 = m3to2_plus + m3to2_minus

            # Compute final face embeddings (equation 11 in the main manuscript)
            h2pp = self.__mlp_processor_2p(torch.cat([h2p, m3to2], dim=1))

            # Propagate faces data to the objects (equation 12 in the main manuscript)
            m2to4 = self.__mlp_processor_2to4(h2pp)
            m2to4 = propagate_messages(m2to4, b24, reduce="mean")
            h4p = self.__mlp_processor_4(torch.cat((h4, m2to4), dim=1))
        else:
            h2pp = h2 
            h3p_minus = h3_minus
            h3p_plus = h3_plus
            h4p = h4

        # STEP 1: Update nodes and objects (equation 13 in the main manuscript)
        m0to4 = self.__mlp_processor_0to4(h0)
        m4to0 = self.__mlp_processor_4to0(h4p)

        m0to4 = propagate_messages(m0to4, b04, reduce="mean")
        m4to0 = propagate_messages(m4to0, b04.T)

        h0p = self.__mlp_processor_0(torch.cat((h0, m4to0), dim=1)) # (equation 14 in the main manuscript)
        h4pp = self.__mlp_processor_4p(torch.cat((h4p, m0to4), dim=1)) # (equation 15 in the main manuscript)

        return (
            h0p,
            h1,
            h2pp,
            h3p_minus,
            h3p_plus,
            h4pp,
            b02,
            b04,
            b12,
            b23,
            b24,
        )


class HOPNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        hid_channels: list[int],
        num_layers: int,
        activation_func: Callable,
        mlp_layers: int,
        out_channels: list[int],
    ):
        """Higher-Order topological Physics-informed Network

        Parameters
        ----------
        in_channels : list[int]
            Number of input channels for [0,1,2,3,4]-rank cells
        hid_channels : list[int]
            Number of hidden channels for processing [0,1,2,3,4]-rank cells
        num_layers : int
            Number of independent message-passing layers
        activation_func : Callable
            Pytorch activation function (e.g. torch.nn.ReLU, torch.nn.SELU)
        mlp_layers : int
            Number of linear layers in MLPs
        out_channels : list[int]
            Number of output channels for [0,1,2,3,4]-rank cells
        """
        super().__init__()

        assert len(in_channels) == 5  # x0, x1, x2, x3, x4
        assert len(hid_channels) == 5  # x0, x1, x2, x3, x4
        assert len(out_channels) == 2  # x0, x4
        assert num_layers > 0
        assert mlp_layers > 0

        self.__x3_encoding_indices_minus = list(range(0, 28))
        self.__x3_encoding_indices_plus = (
            list(range(0, 4)) + list(range(16, 28)) + list(range(4, 16))
        )

        # Step 1: Encoders
        self.__mlp_encoder_0 = mlp(in_channels[0], hid_channels[0], hid_channels[0], activation_func, mlp_layers)
        self.__mlp_encoder_1 = mlp(in_channels[1], hid_channels[1], hid_channels[0], activation_func, mlp_layers)
        self.__mlp_encoder_2 = mlp(in_channels[2], hid_channels[2], hid_channels[2], activation_func, mlp_layers)
        self.__mlp_encoder_3 = mlp(in_channels[3], hid_channels[3], hid_channels[3], activation_func, mlp_layers)
        self.__mlp_encoder_4 = mlp(in_channels[4], hid_channels[4], hid_channels[4], activation_func, mlp_layers)

        # Step 2: Processors & Message-Passing
        layers: list = []
        for _ in range(num_layers):
            layers.append(HOPNet_Layer(hid_channels, activation_func, mlp_layers))
        self.__layers = torch.nn.Sequential(*layers)

        # Step 3: Decoders
        self.__mlp_decoder_0 = mlp(hid_channels[0], hid_channels[0], out_channels[0], activation_func, mlp_layers, False, False)
        self.__mlp_decoder_4 = mlp(hid_channels[4], hid_channels[4], out_channels[1], activation_func, mlp_layers, False, False)

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor | None,
        x3: torch.Tensor | None,
        x4: torch.Tensor,
        t0_zero_acc: torch.Tensor,
        t4_zero_acc: torch.Tensor,
        b02: torch.Tensor | None,
        b04: torch.Tensor,
        b12: torch.Tensor | None,
        b23: torch.Tensor | None,
        b24: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of HOPNet

        Parameters
        ----------
        x0 : torch.Tensor
            Nodes (0-rank) input features
        x1 : torch.Tensor
            Edges (1-rank) input features
        x2 : torch.Tensor
            Faces (2-rank) input features
        x3 : torch.Tensor
            Collisions (3-rank) input features
        x4 : torch.Tensor
            Objects (4-rank) input features
        t0_zero_acc : torch.Tensor
            Node acceleration value equivalent to 0 acceleration (normalization mean)
        t4_zero_acc : torch.Tensor
            Object acceleration value equivalent to 0 acceleration (normalization mean)
        b02 : torch.Tensor
            Upper-incidence matrix from nodes to faces
        b04 : torch.Tensor
            Upper-incidence matrix from nodes to objects
        b12 : torch.Tensor
            Upper-incidence matrix from edges to faces
        b23 : torch.Tensor
            Upper-incidence matrix from faces to collision faces
        b24 : torch.Tensor
            Upper-incidence matrix from faces to objects

        Returns
        -------
        torch.Tensor
            Final hidden states of the nodes cells (0-rank) (nodes_num, out_channels)
        torch.Tensor
            Final hidden states of the objects cells (4-rank) (objects_num, out_channels)
        """
        # Step 0: Get objects type (static = 0; dynamic = 1) to force static acc to 0
        # Check if < 0 (because after normalization, static is [-1., 1] and dynamic [1, -1])
        static_objs_idx = (x4[:, 8] < 0).nonzero()

        # Step 1: Encode the inputs (equation 7 in the main manuscript)
        h0 = self.__mlp_encoder_0(x0)  # (nodes_count, out_c)
        h1 = self.__mlp_encoder_1(x1)  # (edges_count, out_c)
        h2 = self.__mlp_encoder_2(x2) if x2 is not None else None  # (faces_count, )
        h4 = self.__mlp_encoder_4(x4)  # (obj_count, out_c)

        # Special dual encoding of x3 (for permutation invariance in collisions)
        if x3 is not None:
            h3_minus = self.__mlp_encoder_3(x3[:, self.__x3_encoding_indices_minus])
            x3[:, 0:3] *= -1.0  # Negate the collision vector (but not the norm)
            h3_plus = self.__mlp_encoder_3(x3[:, self.__x3_encoding_indices_plus])
        else:
            h3_minus = None
            h3_plus = None

        # Step 2: Process (message-passing layers)
        out = self.__layers(
            (h0, h1, h2, h3_minus, h3_plus, h4, b02, b04, b12, b23, b24)
        )

        # Step 3: Decoding (equation 16 in the main manuscript)
        acc_nodes = self.__mlp_decoder_0(out[0]) # out[0] = h0p
        acc_objs = self.__mlp_decoder_4(out[5])  # out[5] = h4pp

        # Force static objects acceleration to 0
        acc_objs[static_objs_idx, :] = t4_zero_acc
        # Force static nodes acceleration to 0 (using b04)
        _, objects = b04.indices()
        static_nodes_idx = (objects == static_objs_idx).nonzero()[:, 1]
        acc_nodes[static_nodes_idx] = t0_zero_acc

        return acc_nodes, acc_objs
