# File:         ablations.py
# Date:         2024/11/12
# Description:  Ablation study variants of HOPNet

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


class HOPNet_NoSequential_Layer(torch.nn.Module):
    """Single Message-Passing layer for ablated HOPNet without sequential message-passing"""

    def __init__(self, channels: list[int], activation_func: Callable, mlp_layers: int):
        super().__init__()
        assert len(channels) == 5  # x0, x1, x2, x3, x4
        self.__channels = channels

        # 0-cells
        self.__mlp_processor_0to0 = mlp(channels[0], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_0to1 = mlp(channels[0], channels[1], channels[1], activation_func, mlp_layers)
        self.__mlp_processor_0to2 = mlp(channels[0], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_0to3 = mlp(channels[0], channels[3], channels[3], activation_func, mlp_layers)
        self.__mlp_processor_0to4 = mlp(channels[0], channels[4], channels[4], activation_func, mlp_layers)

        # 1-cells
        self.__mlp_processor_1to0 = mlp(channels[1], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_1to1 = mlp(channels[1], channels[1], channels[1], activation_func, mlp_layers)
        self.__mlp_processor_1to2 = mlp(channels[1], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_1to3 = mlp(channels[1], channels[3], channels[3], activation_func, mlp_layers)
        self.__mlp_processor_1to4 = mlp(channels[1], channels[4], channels[4], activation_func, mlp_layers)

        # 2-cells
        self.__mlp_processor_2to0 = mlp(channels[2], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_2to1 = mlp(channels[2], channels[1], channels[1], activation_func, mlp_layers)
        self.__mlp_processor_2to2 = mlp(channels[2], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_2to3 = mlp(channels[2], channels[3], channels[3], activation_func, mlp_layers)
        self.__mlp_processor_2to4 = mlp(channels[2], channels[4], channels[4], activation_func, mlp_layers)

        # 3-cells (m33 and m34 do not exist because physically impossible)
        self.__mlp_processor_3to0 = mlp(channels[3], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_3to1 = mlp(channels[3], channels[1], channels[1], activation_func, mlp_layers)
        self.__mlp_processor_3to2 = mlp(channels[3], channels[2], channels[2], activation_func, mlp_layers)

        # 4-cells (m43 and m44 do not exist because physically impossible)
        self.__mlp_processor_4to0 = mlp(channels[4], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_4to1 = mlp(channels[4], channels[1], channels[1], activation_func, mlp_layers)
        self.__mlp_processor_4to2 = mlp(channels[4], channels[2], channels[2], activation_func, mlp_layers)

        # Aggregation networks
        self.__mlp_processor_0 = mlp(6 * channels[0], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_1 = mlp(6 * channels[1], channels[1], channels[1], activation_func, mlp_layers)
        self.__mlp_processor_2 = mlp(6 * channels[2], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_3 = mlp(4 * channels[3], channels[3], channels[3], activation_func, mlp_layers)
        self.__mlp_processor_4 = mlp(4 * channels[4], channels[4], channels[4], activation_func, mlp_layers)

    def forward(self, x: tuple) -> tuple:
        (
            h0,
            h1,
            h2,
            h3_minus,
            h3_plus,
            h4,
            a010,
            a101,
            a232,
            b01,
            b02,
            b03,
            b04,
            b12,
            b13,
            b14,
            b23,
            b24,
            m2to0,
            m2to1,
            m2to4,
        ) = x

        # Messages from nodes
        m0to0 = self.__mlp_processor_0to0(h0)
        m0to1 = self.__mlp_processor_0to1(h0)
        m0to4 = self.__mlp_processor_0to4(h0)
        # Messages from edges
        m1to0 = self.__mlp_processor_1to0(h1)
        m1to1 = self.__mlp_processor_1to1(h1)
        m1to4 = self.__mlp_processor_1to4(h1)
        # Messages from objects
        m4to0 = self.__mlp_processor_4to0(h4)
        m4to1 = self.__mlp_processor_4to1(h4)

        # Message-passing
        m0to0 = propagate_messages(m0to0, a010)
        m0to1 = propagate_messages(m0to1, b01)
        m0to4 = propagate_messages(m0to4, b04, reduce="mean")
        m1to0 = propagate_messages(m1to0, b01.T)
        m1to1 = propagate_messages(m1to1, a101)
        m1to4 = propagate_messages(m1to4, b14, reduce="mean")
        m4to0 = propagate_messages(m4to0, b04.T)
        m4to1 = propagate_messages(m4to1, b14.T)

        # Compute only if collision cells are defined
        if h3_minus is not None:
            # Messages from nodes
            m0to2 = self.__mlp_processor_0to2(h0)
            m0to3 = self.__mlp_processor_0to3(h0)
            # Messages from edges
            m1to2 = self.__mlp_processor_1to2(h1)
            m1to3 = self.__mlp_processor_1to3(h1)
            # Messages from faces
            m2to0 = self.__mlp_processor_2to0(h2)
            m2to1 = self.__mlp_processor_2to1(h2)
            m2to2 = self.__mlp_processor_2to2(h2)
            m2to3 = self.__mlp_processor_2to3(h2)
            m2to4 = self.__mlp_processor_2to4(h2)
            # Messages from collision contacts
            m3to0_plus = self.__mlp_processor_3to0(h3_plus)
            m3to0_minus = self.__mlp_processor_3to0(h3_minus)
            m3to1_plus = self.__mlp_processor_3to1(h3_plus)
            m3to1_minus = self.__mlp_processor_3to1(h3_minus)
            m3to2_plus = self.__mlp_processor_3to2(h3_plus)
            m3to2_minus = self.__mlp_processor_3to2(h3_minus)
            # Messages from objects
            m4to2 = self.__mlp_processor_4to2(h4)

            # Message-passing
            m0to2 = propagate_messages(m0to2, b02)
            m0to3 = propagate_messages(m0to3, b03)
            m1to2 = propagate_messages(m1to2, b12)
            m1to3 = propagate_messages(m1to3, b13)
            m2to0 = propagate_messages(m2to0, b02.T)
            m2to1 = propagate_messages(m2to1, b12.T)
            m2to2 = propagate_messages(m2to2, a232)
            m2to3 = propagate_messages(m2to3, b23)
            m2to4 = propagate_messages(m2to4, b24)
            m3to0_plus = propagate_messages(m3to0_plus, b03.T)
            m3to0_minus = propagate_messages(m3to0_minus, b03.T)
            m3to1_plus = propagate_messages(m3to1_plus, b13.T)
            m3to1_minus = propagate_messages(m3to1_minus, b13.T)
            m3to2_plus = propagate_messages(m3to2_plus, b23.T)
            m3to2_minus = propagate_messages(m3to2_minus, b23.T)
            m4to2 = propagate_messages(m4to2, b24.T)

            h3p_minus = self.__mlp_processor_3(torch.cat((h3_minus, m0to3, m1to3, m2to3), dim=1))
            h3p_plus = self.__mlp_processor_3(torch.cat((h3_plus, m0to3, m1to3, m2to3), dim=1))
            h2p = self.__mlp_processor_2(
                torch.cat((h2, m0to2, m1to2, m2to2, m3to2_plus + m3to2_minus, m4to2), dim=1)
            )
        else:
            h3p_minus = h3_minus
            h3p_plus = h3_plus
            h2p = h2
            # No collisions => no faces, no collisions
            m3to0_plus = m2to0
            m3to0_minus = m3to0_plus
            m3to1_plus = m2to1
            m3to1_minus = m3to1_plus

        # Message-aggregation
        h0p = self.__mlp_processor_0(
            torch.cat((h0, m0to0, m1to0, m2to0, m3to0_plus + m3to0_minus, m4to0), dim=1)
        )
        h1p = self.__mlp_processor_1(
            torch.cat((h1, m0to1, m1to1, m2to1, m3to1_plus + m3to1_minus, m4to1), dim=1)
        )
        h4p = self.__mlp_processor_4(torch.cat((h4, m0to4, m1to4, m2to4), dim=1))

        return (
            h0p,
            h1p,
            h2p,
            h3p_minus,
            h3p_plus,
            h4p,
            a010,
            a101,
            a232,
            b01,
            b02,
            b03,
            b04,
            b12,
            b13,
            b14,
            b23,
            b24,
            m2to0,
            m2to1,
            m2to4,
        )


class HOPNet_NoSequential(torch.nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        hid_channels: list[int],
        num_layers: int,
        activation_func: Callable,
        mlp_layers: int,
        out_channels: list[int],
    ):
        """Ablated HOPNet without sequential message-passing

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
            Number of output channels for [0,4]-rank cells
        """
        super().__init__()

        assert len(in_channels) == 5  # x0, x1, x2, x3, x4
        assert len(hid_channels) == 5  # x0, x1, x2, x3, x4
        assert len(out_channels) == 2  # x0, x4
        assert num_layers > 0
        assert mlp_layers > 0

        self.__hid_channels = hid_channels
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
            layers.append(HOPNet_NoSequential_Layer(hid_channels, activation_func, mlp_layers))
        self.__layers = torch.nn.Sequential(*layers)

        # Step 3: Decoders
        self.__mlp_decoder_0 = mlp(hid_channels[0], hid_channels[0], out_channels[0], activation_func, mlp_layers, False, False)
        self.__mlp_decoder_4 = mlp(hid_channels[4], hid_channels[4], out_channels[1], activation_func, mlp_layers, False, False)

    @property
    def hid_channels(self) -> list[int]:
        return self.__hid_channels

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor | None,
        x3: torch.Tensor | None,
        x4: torch.Tensor,
        t0_zero_acc: torch.Tensor,
        t4_zero_acc: torch.Tensor,
        a010: torch.Tensor,
        a101: torch.Tensor,
        a232: torch.Tensor | None,
        b01: torch.Tensor,
        b02: torch.Tensor | None,
        b03: torch.Tensor | None,
        b04: torch.Tensor,
        b12: torch.Tensor | None,
        b13: torch.Tensor | None,
        b14: torch.Tensor,
        b23: torch.Tensor | None,
        b24: torch.Tensor | None,
        m20: torch.Tensor,
        m21: torch.Tensor,
        m24: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x0 : torch.Tensor
            Input features on nodes.
        x1 : torch.Tensor
            Input features on edges.
        x2 : torch.Tensor
            Input features on faces.
        x3 : torch.Tensor
            Input features on collision cells.
        x4 : torch.Tensor
            Input features on objects.
        t0_zero_acc : torch.Tensor
            Node acceleration value equivalent to 0 acceleration (normalization mean)
        t4_zero_acc : torch.Tensor
            Object acceleration value equivalent to 0 acceleration (normalization mean)
        a010: torch.Tensor
        a101 : torch.Tensor
        a232 : torch.Tensor
        b01 : torch.Tensor
        b02 : torch.Tensor
        b03 : torch.Tensor
        b04 : torch.Tensor
        b12 : torch.Tensor
        b13 : torch.Tensor
        b14 : torch.Tensor
        b23: torch.Tensor
        b24: torch.Tensor
        m20 : torch.Tensor
        m21 : torch.Tensor
        m24 : torch.Tensor

        Returns
        -------
        torch.Tensor, shape = (n_nodes, out_channels_0)
            Final hidden states of the nodes (0-cells).
        torch.Tensor, shape = (n_obj, out_channels_2)
            Final hidden states of the objects (4-cells).
        """
        # Step 0: Get objects type (static = 0; dynamic = 1) to force static acc to 0
        # Check if < 0 (because after normalization, static is [-1., 1] and dynamic [1, -1])
        static_objs_idx = (x4[:, 8] < 0).nonzero()

        # Step 1: Encode the inputs
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
            (
                h0,
                h1,
                h2,
                h3_minus,
                h3_plus,
                h4,
                a010,
                a101,
                a232,
                b01,
                b02,
                b03,
                b04,
                b12,
                b13,
                b14,
                b23,
                b24,
                m20,
                m21,
                m24,
            )
        )

        # Step 3: Decoding
        acc_nodes = self.__mlp_decoder_0(out[0])
        acc_objs = self.__mlp_decoder_4(out[5])

        # Force static objects acceleration to 0
        acc_objs[static_objs_idx, :] = t4_zero_acc
        # Force static nodes acceleration to 0 (using b04)
        _, objects = b04.indices()
        static_nodes_idx = (objects == static_objs_idx).nonzero()[:, 1]
        acc_nodes[static_nodes_idx] = t0_zero_acc

        return acc_nodes, acc_objs


class HOPNet_NoObjcetCells_Layer(torch.nn.Module):
    """Single Message-Passing layer for ablated HOPNet without 4-rank objet cells"""

    def __init__(self, channels: list[int], activation_func: Callable, mlp_layers: int):
        super().__init__()
        assert len(channels) == 4  # x0, x1, x2, x3
        self.__x2_channels = channels[2]

        # CASE 0: No Collision

        # Step 1: Enhancement of face embeddings
        self.__mlp_processor_0to2 = mlp(channels[0], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_1to2 = mlp(channels[1], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_2 = mlp(3 * channels[2], channels[2], channels[2], activation_func, mlp_layers)

        # Steps 2 and 3 : Inter-objects collisions
        self.__mlp_processor_2to3 = mlp(channels[2], channels[3], channels[3], activation_func, mlp_layers)
        self.__mlp_processor_3 = mlp(3 * channels[2], channels[2], channels[2], activation_func, mlp_layers)
        self.__mlp_processor_2p = mlp(2 * channels[2], channels[2], channels[2], activation_func, mlp_layers)

        # Step 4: Intra-object node update
        self.__mlp_processor_2to0 = mlp(channels[2], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_0 = mlp(2 * channels[0], channels[0], channels[0], activation_func, mlp_layers)

        # Step 5: Additional node-to-node message-passing
        self.__mlp_processor_0to0 = mlp(channels[0], channels[0], channels[0], activation_func, mlp_layers)
        self.__mlp_processor_0p = mlp(2 * channels[0], channels[0], channels[0], activation_func, mlp_layers)

    def forward(self, x: tuple) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        (h0, h1, h2, h3_minus, h3_plus, a010, b02, b12, b23) = x
        # STEP 0: Process collisions (if any)
        if h3_minus is not None:
            assert h3_plus is not None
            assert h2 is not None
            assert b23 is not None
            assert b12 is not None

            # Compute messages to build up faces (equation 8 in the main manuscript)
            m0to2 = self.__mlp_processor_0to2(h0)
            m1to2 = self.__mlp_processor_1to2(h1)

            # Create new face embedding (equation 9 in the main manuscript without object cells)
            m0to2 = propagate_messages(m0to2, b02)
            m1to2 = propagate_messages(m1to2, b12)
            h2p = self.__mlp_processor_2(torch.cat((h2, m0to2, m1to2), dim=1))

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

            # Propagate faces data to the objects
            m2to0 = self.__mlp_processor_2to0(h2pp)
            m2to0 = propagate_messages(m2to0, b02.T)
            h0p = self.__mlp_processor_0(torch.cat((h0, m2to0), dim=1))
        else:
            h0p = h0
            h2pp = h2
            h3p_minus = h3_minus
            h3p_plus = h3_plus

        # STEP 1: Update nodes and objects (step not present in HOPNet)
        m0to0 = self.__mlp_processor_0to0(h0p)
        m0to0 = propagate_messages(m0to0, a010)
        h0pp = self.__mlp_processor_0p(torch.cat((h0p, m0to0), dim=1))

        return h0pp, h1, h2pp, h3p_minus, h3p_plus, a010, b02, b12, b23


class HOPNet_NoObjectCells(torch.nn.Module):

    def __init__(
        self,
        in_channels: list[int],
        hid_channels: list[int],
        num_layers: int,
        activation_func: Callable,
        mlp_layers: int,
        out_channels: int,
    ):
        """Ablated HOPNet without 4-rank object cells

        Parameters
        ----------
        in_channels : list[int]
            Number of input channels for [0,1,2,3]-rank cells
        hid_channels : list[int]
            Number of hidden channels for processing [0,1,2,3]-rank cells
        num_layers : int
            Number of independent message-passing layers
        activation_func : Callable
            Pytorch activation function (e.g. torch.nn.ReLU, torch.nn.SELU)
        mlp_layers : int
            Number of linear layers in MLPs
        out_channels : int
            Number of output channels for [0-rank cells
        """
        super().__init__()

        assert len(in_channels) == 4  # x0, x1, x2, x3
        assert len(hid_channels) == 4  # x0, x1, x2, x3
        assert num_layers > 0
        assert mlp_layers > 0

        self.__hid_channels = hid_channels
        self.__x3_encoding_indices_minus = list(range(0, 28))
        self.__x3_encoding_indices_plus = (
            list(range(0, 4)) + list(range(16, 28)) + list(range(4, 16))
        )

        # Step 1: Encoders
        self.__mlp_encoder_0 = mlp(in_channels[0], hid_channels[0], hid_channels[0], activation_func, mlp_layers)
        self.__mlp_encoder_1 = mlp(in_channels[1], hid_channels[1], hid_channels[0], activation_func, mlp_layers)
        self.__mlp_encoder_2 = mlp(in_channels[2], hid_channels[2], hid_channels[2], activation_func, mlp_layers)
        self.__mlp_encoder_3 = mlp(in_channels[3], hid_channels[3], hid_channels[3], activation_func, mlp_layers)

        # Step 2: Processors & Message-Passing
        layers: list = []
        for _ in range(num_layers):
            layers.append(HOPNet_NoObjcetCells_Layer(hid_channels, activation_func, mlp_layers))
        self.__layers = torch.nn.Sequential(*layers)

        # Step 3: Decoders
        self.__mlp_decoder_0 = mlp(hid_channels[0], hid_channels[0], out_channels, activation_func, mlp_layers, False, False)

    @property
    def hid_channels(self) -> list[int]:
        return self.__hid_channels

    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor | None,
        x3: torch.Tensor | None,
        t0_zero_acc: torch.Tensor,
        a010: torch.Tensor,
        b02: torch.Tensor | None,
        b12: torch.Tensor | None,
        b23: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x0 : torch.Tensor
            Input features on nodes.
        x1 : torch.Tensor
            Input features on edges.
        x2 : torch.Tensor
            Input features on faces.
        x3 : torch.Tensor
            Input features on collision cells.
        t0_zero_acc : torch.Tensor
            Node acceleration value equivalent to 0 acceleration (normalization mean)
        a010: torch.Tensor
        b02 : torch.Tensor
        b12 : torch.Tensor
        b23: torch.Tensor

        Returns
        -------
        torch.Tensor, shape = (n_nodes, out_channels)
            Final hidden states of the nodes (0-cells).
        """
        # Step 0: Get objects type (static = 0; dynamic = 1) to force static acc to 0
        # Check if < 0 (because after normalization, static is [-1., 1] and dynamic [1, -1])
        static_nodes_idx = (x0[:, 16] < 0).nonzero()

        # Step 1: Encode the inputs (equation 7 in the main manuscript)
        h0 = self.__mlp_encoder_0(x0)  # (nodes_count, out_c)
        h1 = self.__mlp_encoder_1(x1)  # (edges_count, out_c)
        h2 = self.__mlp_encoder_2(x2) if x2 is not None else None  # (faces_count, )

        # Special dual encoding of x3 (for permutation invariance in collisions)
        if x3 is not None:
            h3_minus = self.__mlp_encoder_3(x3[:, self.__x3_encoding_indices_minus])
            x3[:, 0:3] *= -1.0  # Negate the collision vector (but not the norm)
            h3_plus = self.__mlp_encoder_3(x3[:, self.__x3_encoding_indices_plus])
        else:
            h3_minus = None
            h3_plus = None

        # Step 2: Process (message-passing layers)
        out = self.__layers((h0, h1, h2, h3_minus, h3_plus, a010, b02, b12, b23))

        # Step 3: Decoding (equation 16 in the main manuscript, only for nodes)
        acc_nodes = self.__mlp_decoder_0(out[0])

        # Force static nodes acceleration to 0 (using b04)
        acc_nodes[static_nodes_idx] = t0_zero_acc

        return acc_nodes
