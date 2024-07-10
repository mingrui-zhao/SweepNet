"""Model architectures."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from dgcnn import DGCNNFeat


class VoxelEncoder(nn.Module):
    """Input voxel encoder ."""

    def __init__(self, ef_dim=32):
        super(VoxelEncoder, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(1, self.ef_dim, 4, stride=2, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(
            self.ef_dim, self.ef_dim * 2, 4, stride=2, padding=1, bias=True
        )
        self.conv_3 = nn.Conv3d(
            self.ef_dim * 2, self.ef_dim * 4, 4, stride=2, padding=1, bias=True
        )
        self.conv_4 = nn.Conv3d(
            self.ef_dim * 4, self.ef_dim * 8, 4, stride=2, padding=1, bias=True
        )
        self.conv_5 = nn.Conv3d(
            self.ef_dim * 8, self.ef_dim * 8, 4, stride=1, padding=0, bias=True
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for conv in [self.conv_1, self.conv_2, self.conv_3, self.conv_4, self.conv_5]:
            nn.init.xavier_uniform_(conv.weight)
            nn.init.constant_(conv.bias, 0)

    def forward(self, inputs):
        d = inputs
        for i in range(1, 6):
            d = getattr(self, f"conv_{i}")(d)
            d = F.leaky_relu(d, negative_slope=0.01)
        d = d.view(-1, self.ef_dim * 8)
        d = F.leaky_relu(d, negative_slope=0.01)
        return d


class NeuralSweeper(nn.Module):
    """Neural sweeper module."""

    def __init__(
        self,
        num_primitives,
        sharpness,
        bspline_control_points,
        neural_sweeper,
        bspline_cache,
    ):
        super(NeuralSweeper, self).__init__()
        self.num_primitives = num_primitives
        self.sharpness = sharpness
        self.bspline_control_points = bspline_control_points
        self.neural_sweeper = neural_sweeper
        self.union_mask = nn.Parameter(torch.ones(1, self.num_primitives)).cuda()
        self.bspline_cache = bspline_cache

    def forward(
        self,
        sample_point_coordinates,  # B x N x 3
        primitive_parameters,  # B x K x 29
        union_layer_weights,  # B x 2 x num_intersections
        is_training,  # bool
    ):
        """Occupancy querying with the predicted parameters."""
        # B is batch size, K is number of primitives
        B, K, _ = (
            primitive_parameters.shape
        )  # Batch size, num primitives, number of parameters

        _, N, _ = sample_point_coordinates.shape  # Batch size, number of points, 3

        # poco model
        data = {}
        bspline_points, _, profile_points = utils.sample_scaling_superellipse_points(
            primitive_parameters,
            num_loops=15,
            loop_points=50,
            n=self.bspline_control_points,
            bspline_cache=self.bspline_cache,
        )
        manifold_points = torch.cat([bspline_points, profile_points], dim=1)
        manifold_points = manifold_points.permute(0, 2, 1)
        sample_point_coordinates = sample_point_coordinates.permute(
            0, 2, 1
        ).repeat_interleave(K, dim=0)
        data["x"] = torch.ones_like(manifold_points)
        data["pos"] = manifold_points
        data["pos_non_manifold"] = sample_point_coordinates
        data["occupancies"] = None
        occupancy_pre_selection = (
            torch.nn.Sigmoid()(self.neural_sweeper(data, spectral_only=False))
            .reshape(B, K, N)
            .permute(0, 2, 1)
        )  # B x N x K
        occupancy_pre_union = torch.einsum(
            "bc,bmc->bmc", union_layer_weights, occupancy_pre_selection
        )
        if not is_training:
            occupancies = torch.max(occupancy_pre_union, dim=-1)[0]
        else:
            with torch.no_grad():
                # Boltzmann operator
                weights = torch.softmax(occupancy_pre_union * (40), dim=-1)
            occupancies = torch.sum(weights * occupancy_pre_union, dim=-1)
        return (
            occupancies,
            occupancy_pre_union,
            occupancy_pre_selection,
            union_layer_weights,
        )


class SelectionHead(nn.Module):
    """Selection head that select primitives."""

    def __init__(self, feature_dim, num_primitives):
        super(SelectionHead, self).__init__()
        self.num_primitives = num_primitives
        self.feature_dim = feature_dim
        self.union_linear = nn.Linear(
            self.feature_dim * 8, self.num_primitives, bias=True
        )

    def forward(self, feature, is_training):
        # getting union layer connection weights
        union_layer_weights = self.union_linear(feature)
        union_layer_weights = union_layer_weights.view(
            -1, self.num_primitives
        )  # [B,c_dim]

        if not is_training:
            union_layer_weights = (union_layer_weights > 0).type(torch.float32)
        else:
            # during train, we use continues connection weights to get better gradients
            union_layer_weights = torch.sigmoid(union_layer_weights)

        return union_layer_weights


class SweptVolumeHead(nn.Module):
    """Learning module that predicts primitive paramters."""

    def __init__(
        self,
        feature_dim,
        num_primitives,
        bspline_control_points,
    ):
        super(SweptVolumeHead, self).__init__()
        self.num_primitives = num_primitives
        self.feature_dim = feature_dim
        self.bspline_control_points = bspline_control_points
        self.num_primitive_parameters = bspline_control_points * 3 + 3 + 2
        self.primitive_linear = nn.Linear(
            self.feature_dim * 8,
            self.num_primitives * self.num_primitive_parameters,
            bias=True,
        )
        nn.init.xavier_uniform_(self.primitive_linear.weight)
        nn.init.constant_(self.primitive_linear.bias, 0)

        # Rectify sweep surface parameter range
        shapes_adder = torch.zeros(
            1, self.num_primitives, self.num_primitive_parameters
        )
        shapes_adder[:, :, : self.bspline_control_points * 3] = -0.5
        shapes_adder[:, :, self.bspline_control_points * 3 + 3 :] = -0.5
        shapes_adder[
            :, :, self.bspline_control_points * 3 : self.bspline_control_points * 3 + 2
        ] = 0.01
        shapes_adder[:, :, self.bspline_control_points * 3 + 2] = 0.5
        self.shapes_adder = shapes_adder.to("cuda")

        shapes_multiplier = torch.ones(
            1, self.num_primitives, self.num_primitive_parameters
        )
        shapes_multiplier[
            :, :, self.bspline_control_points * 3 : self.bspline_control_points * 3 + 2
        ] = 0.49
        shapes_multiplier[:, :, self.bspline_control_points * 3 + 2] = 4.5
        self.shapes_multiplier = shapes_multiplier.to("cuda")

    def forward(self, feature):
        # One linear layer that decodes features to primitive parameters
        shapes = self.primitive_linear(feature)
        shapes = torch.sigmoid(shapes)
        shapes = shapes.view(-1, self.num_primitives, self.num_primitive_parameters)
        shapes = shapes * self.shapes_multiplier + self.shapes_adder
        return shapes


class Decoder(nn.Module):
    # Decoder, a multilayer MLP to decode the feature extracted by DGCNN
    def __init__(self, feature_dim):
        # Initialise the neurons with xavier initilisation for gradient control.
        super(Decoder, self).__init__()
        self.feature_dim = feature_dim
        self.linear_1 = nn.Linear(self.feature_dim, self.feature_dim * 2, bias=True)
        self.linear_2 = nn.Linear(self.feature_dim * 2, self.feature_dim * 4, bias=True)
        self.linear_3 = nn.Linear(self.feature_dim * 4, self.feature_dim * 8, bias=True)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.constant_(self.linear_1.bias, 0)
        nn.init.xavier_uniform_(self.linear_2.weight)
        nn.init.constant_(self.linear_2.bias, 0)
        nn.init.xavier_uniform_(self.linear_3.weight)
        nn.init.constant_(self.linear_3.bias, 0)

    def forward(self, inputs):
        # THre layer MLP with leaky relu activation
        l1 = self.linear_1(inputs)
        l1 = F.leaky_relu(l1, negative_slope=0.01, inplace=True)
        l2 = self.linear_2(l1)
        l2 = F.leaky_relu(l2, negative_slope=0.01, inplace=True)
        l3 = self.linear_3(l2)
        l3 = F.leaky_relu(l3, negative_slope=0.01, inplace=True)
        return l3


class SweepNet(nn.Module):
    """SweepNet network."""

    def __init__(self, config, neural_sweeper, bspline_cache=None):
        super(SweepNet, self).__init__()
        self.config = config
        self.neural_sweeper = neural_sweeper
        self.num_primitives = self.config.num_primitives
        self.feature_dim = self.config.feature_dim
        self.sharpness = self.config.sharpness
        self.bspline_control_points = self.config.bspline_control_points

        self.encoder = VoxelEncoder()

        # Decoder is an MLP
        self.decoder = Decoder(self.feature_dim)

        # CSGStump Connection Head
        self.selection_head = SelectionHead(self.feature_dim, self.num_primitives)
        self.bspline_cache = bspline_cache

        self.swept_volume_head = SweptVolumeHead(
            self.feature_dim,
            self.num_primitives,
            self.bspline_control_points,
        )

        self.neural_sweeper = NeuralSweeper(
            self.num_primitives,
            self.sharpness,
            self.bspline_control_points,
            self.neural_sweeper,
            self.bspline_cache,
        )

    def forward(self, voxel, sample_coordinates, is_training=True):
        # First encode voxel
        feature = self.encoder(voxel)
        # This is feature enhancer by propagating feature dimensions
        code = self.decoder(feature)
        # Find connections
        union_layer_connections = self.selection_head(code, is_training=is_training)
        # Parametrise primitives
        primitive_parameters = self.swept_volume_head(code)
        # Query occupancy
        (
            occupancies,
            occupancies_pre_union,
            occupancies_pre_selection,
            union_mask,
        ) = self.neural_sweeper(
            sample_coordinates,
            primitive_parameters,
            union_layer_connections,
            is_training=is_training,
        )
        return (
            occupancies,
            occupancies_pre_union,
            primitive_parameters,
            union_layer_connections,
        )


class SweepNetPCD(nn.Module):
    def __init__(self, config, neural_sweeper, bspline_cache=None):
        super(SweepNetPCD, self).__init__()
        self.config = config
        self.neural_sweeper = neural_sweeper
        self.num_primitives = self.config.num_primitives
        self.feature_dim = self.config.feature_dim
        self.sharpness = self.config.sharpness
        self.bspline_control_points = self.config.bspline_control_points

        self.encoder = DGCNNFeat(global_feat=True)

        # Decoder is an MLP
        self.decoder = Decoder(self.feature_dim)

        # CSGStump Connection Head
        self.selection_head = SelectionHead(self.feature_dim, self.num_primitives)
        self.bspline_cache = bspline_cache

        self.swept_volume_head = SweptVolumeHead(
            self.feature_dim,
            self.num_primitives,
            self.bspline_control_points,
        )

        self.neural_sweeper = NeuralSweeper(
            self.num_primitives,
            self.sharpness,
            self.bspline_control_points,
            self.neural_sweeper,
            self.bspline_cache,
        )

    def forward(self, voxel, sample_coordinates, is_training=True):
        # First encode voxel
        feature = self.encoder(voxel)
        # This is feature enhancer by propagating feature dimensions
        code = self.decoder(feature)
        # Find connections
        union_layer_connections = self.selection_head(code, is_training=is_training)
        # Parametrise primitives
        primitive_parameters = self.swept_volume_head(code)
        # Query occupancy
        (
            occupancies,
            occupancies_pre_union,
            occupancies_pre_selection,
            union_mask,
        ) = self.neural_sweeper(
            sample_coordinates,
            primitive_parameters,
            union_layer_connections,
            is_training=is_training,
        )
        return (
            occupancies,
            occupancies_pre_union,
            primitive_parameters,
            union_layer_connections,
        )
