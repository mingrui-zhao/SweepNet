"""All loss functions defined here."""

# pylint: disable=W0613
import torch
import torch.nn as nn
import utils


def chamfer_distance_3d(a, b):
    """
    Compute the Chamfer Distance between two sets of points a and B in 3D.

    A and B are tensors of shape (batch_size, num_points, 3).
    """

    # Compute pairwise distances
    a_expanded = a.unsqueeze(2)
    b_expanded = b.unsqueeze(1)
    distances = torch.norm(a_expanded - b_expanded, dim=3)

    # Find the closest distance from points in A to B and vice versa
    min_distances_a = torch.min(distances, dim=2)[0]
    min_distances_b = torch.min(distances, dim=1)[0]

    # Compute the Chamfer Distance
    chamfer_a = torch.mean(min_distances_a, dim=1)
    chamfer_b = torch.mean(min_distances_b, dim=1)
    return chamfer_a + chamfer_b


def single_way_distance(skeletal_points, bspline_curves):
    # Expand skeletal points to match bspline_curves dimensions for broadcasting
    skeletal_points_expanded = skeletal_points.unsqueeze(
        2
    )  # Adding bspline curve dimension
    distances = torch.norm(
        skeletal_points_expanded - bspline_curves.unsqueeze(1), dim=-1
    )  # Calculate L2 norm across coordinate dimension
    min_distances, _ = distances.min(
        dim=2
    )  # Find minimum distance to any bspline curve for each point
    return min_distances.mean()  # Return the mean of the minimum distances


class BsplineLoss(nn.Module):
    """Loss function for Bspline axis regualrization."""

    def __init__(self, config, bspline_cache):
        super(BsplineLoss, self).__init__()
        self.scale = config.scale_bspline_loss
        self.t_values = torch.linspace(0.0, 0.9999, 16)
        self.n = config.bspline_control_points
        self.k = config.bspline_order
        self.knots = torch.concatenate(
            (
                torch.zeros(self.k),
                torch.linspace(0, 1, self.n - self.k + 1),
                torch.ones(self.k),
            )
        )
        self.config = config
        self.bspline_cache = bspline_cache

    def forward(self, skeletal_points, primitive_parameters):
        """Bslpine loss function."""

        B, N, _ = primitive_parameters.shape
        control_points = primitive_parameters[:, :, : self.n * 3].reshape(-1, self.n, 3)
        bspline_basis = self.bspline_cache.get_bspline_coefficient(
            self.n, self.k, len(self.t_values)
        )
        bspline_curves = torch.matmul(bspline_basis, control_points).reshape(B, -1, 3)
        # loss = single_way_distance(skeletal_points, bspline_curves) * self.scale
        # import pdb; pdb.set_trace()
        loss = chamfer_distance_3d(skeletal_points, bspline_curves) * self.scale
        return loss

class SurfaceLoss(nn.Module):
    """Loss function that compare surface point cloud chamfer distance."""

    def __init__(self, config):
        super(SurfaceLoss, self).__init__()
        self.scale = config.scale_surface_loss

    def forward(self, primitive_parameters, surface_pointcloud, connection_weight):
        """Surface loss function."""
        B = primitive_parameters.shape[0]
        mask = (connection_weight > 0.5).squeeze(0)
        # primitive_parameters = primitive_parameters[:, mask, :]
        _, _, predicted_surface = utils.sample_scaling_superellipse_points(
            primitive_parameters, num_loops=15
        )
        predicted_surface = predicted_surface[mask, ...]
        predicted_surface = predicted_surface.view(B, -1, 3)
        return chamfer_distance_3d(predicted_surface, surface_pointcloud) * self.scale

class ReconLoss(nn.Module):
    """Recnstuction loss determine the reconstruction error."""

    def __init__(self, config):
        super(ReconLoss, self).__init__()
        self.scale = config.scale_recon_loss

    def forward(self, pred_point_value, gt_point_value):
        """Reconstruction loss function."""
        # loss_recon = self.scale * torch.nn.BCELoss()(pred_point_value, gt_point_value)
        loss_recon = self.scale * torch.nn.MSELoss()(pred_point_value, gt_point_value)
        return loss_recon


class ParsimonyLoss(nn.Module):
    """Loss function to encourage sparsity among B-splines."""

    def __init__(self, config):
        super(ParsimonyLoss, self).__init__()
        self.scale = config.scale_parsimony_loss

    def forward(self, weights):
        """Parsimony loss function."""
        loss_parsimony = self.scale * torch.sqrt(torch.sum(torch.sigmoid(weights)))
        return loss_parsimony

class OverlapLoss(nn.Module):
    """Loss function to penalize overlapping primitives."""

    def __init__(self, config):
        super(OverlapLoss, self).__init__()
        self.overlap_threshold = config.overlap_threshold
        self.scale = config.scale_overlap_loss

    def forward(self, occupancy_pre_union):
        """Overlap loss function that enforces each point to be occupied by at most one primitive."""
        B, N, K = occupancy_pre_union.shape

        mask = (occupancy_pre_union > 0.5).float()
        # Calculate the number of primitives occupying each point
        num_primitives_at_point = torch.sum(occupancy_pre_union * mask, dim=2)

        # Calculate penalty for points occupied by more than one primitive
        overlap_penalty = torch.clamp(
            num_primitives_at_point - int(K / self.overlap_threshold), min=0
        )

        # Sum penalties over all points and scale
        loss_overlap = self.scale * overlap_penalty.mean()
        return loss_overlap

class InitLoss(nn.Module):
    """Loss function for Initialization, initialize all primitives near the skeleton."""

    def __init__(self, config):
        super(InitLoss, self).__init__()
        self.scale = config.scale_bspline_loss
        self.config = config
        self.bspline_loss = BsplineLoss(config, bspline_cache=utils.BsplineCache())
        
    def forward(self, skeletal_points, primitive_parameters, union_layer_connections):
        B, N, _ = primitive_parameters.shape
        control_points = primitive_parameters[
            :, :, : self.config.bspline_control_points * 3
        ].reshape(-1, self.config.bspline_control_points, 3)
        loss_bspline = self.bspline_loss(skeletal_points, primitive_parameters)
        #regularise last 5 numbers of primitive_parameters be 0.2, 0.2 ,2, 0., 0.
        dummy_param = torch.tensor([0.05, 0.05, 2., 0., 0.]).to(primitive_parameters.device)
        dummy_param = dummy_param.unsqueeze(0).unsqueeze(0).expand(B, N, 5)
        loss_param = torch.nn.MSELoss()(primitive_parameters[:, :, -5:], dummy_param)
        loss_weight = torch.nn.MSELoss()(union_layer_connections, torch.ones_like(union_layer_connections))
        loss_total = loss_bspline + loss_param + loss_weight
        return {
            "loss_bspline": loss_bspline,
            "loss_param": loss_param,
            "loss_weight": loss_weight,
            "loss_total": loss_total,
        }

class Loss(nn.Module):
    """Loss function for NeuralSweeper."""

    def __init__(self, config):
        super(Loss, self).__init__()
        self.config = config
        self.recon_loss = ReconLoss(config)
        self.bspline_loss = BsplineLoss(config, bspline_cache=utils.BsplineCache())
        self.overlap_loss = OverlapLoss(config)
        self.parsimony_loss = ParsimonyLoss(config)

    def forward(
        self,
        predict_occupancy,
        gt_occupancy,
        occupancy_pre_union,
        primitive_parameters,  # BxNxK
        union_layer_weights,  # BxN # type: ignore
        skeleton_points,  # type: ignore
    ):
        """Loss function forward."""
        loss_recon = self.recon_loss(predict_occupancy, gt_occupancy)
        loss_overlap = self.overlap_loss(occupancy_pre_union)
        primitive_parameters_selected = (
            primitive_parameters * union_layer_weights.unsqueeze(-1)
        )
        loss_bspline = self.bspline_loss(skeleton_points, primitive_parameters_selected)
        loss_parsimony = self.parsimony_loss(union_layer_weights)
        loss_total = loss_recon + loss_bspline + loss_overlap + loss_parsimony

        return {
            "loss_recon": loss_recon,
            "loss_bspline": loss_bspline,
            "loss_parsimony": loss_parsimony,
            "loss_overlap": loss_overlap,
            "loss_total": loss_total,
        }
