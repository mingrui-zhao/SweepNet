"""All loss functions defined here."""

# pylint: disable=W0613
import torch
import torch.nn as nn
import utils
import numpy as np


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


class BranchWiseInitLoss(nn.Module):
    """Loss function for branch-wise initialization, initialize each primitive near its assigned branch."""

    def __init__(self, config):
        super(BranchWiseInitLoss, self).__init__()
        self.scale = config.scale_bspline_loss
        self.config = config
        self.bspline_cache = utils.BsplineCache()
        
        # Pre-compute B-spline basis
        self.n = int(config.bspline_control_points)
        self.k = int(config.bspline_order)
        self.knots = torch.concatenate(
            (
                torch.zeros(self.k),
                torch.linspace(0, 1, self.n - self.k + 1),
                torch.ones(self.k),
            )
        )
        self.t_values = torch.linspace(0.0, 0.9999, 16)
        
    def safe_to_numpy(self, data):
        """
        Safely convert data to numpy array, handling various input types.
        
        Args:
            data: Input data (tensor, numpy array, list, tuple, etc.)
            
        Returns:
            numpy array
        """
        # Handle PyTorch tensors first
        if torch.is_tensor(data):
            return data.detach().cpu().numpy()
        
        # Handle tensor-like objects with cpu() method
        if hasattr(data, 'cpu') and callable(getattr(data, 'cpu')):
            try:
                return data.cpu().numpy()
            except:
                pass
        
        # Handle objects with numpy() method
        if hasattr(data, 'numpy') and callable(getattr(data, 'numpy')):
            try:
                return data.numpy()
            except:
                pass
        
        # Handle numpy arrays (pass through)
        if isinstance(data, np.ndarray):
            return data
        
        # Handle lists/tuples/other iterables
        try:
            return np.array(data)
        except:
            # Last resort: try to convert to float and then to numpy
            try:
                if hasattr(data, '__iter__') and not isinstance(data, (str, bytes)):
                    return np.array([float(x) for x in data])
                else:
                    return np.array([float(data)])
            except:
                # If all else fails, return as single element array
                return np.array([0.0, 0.0, 0.0])  # Default 3D point

    def forward(self, skeletal_points, primitive_parameters, union_layer_connections, 
                branch_assignments, primitive_control_points, branches):
        """
        Forward pass for branch-wise initialization loss.
        
        Args:
            skeletal_points: Full skeleton point cloud (not used in branch-wise loss)
            primitive_parameters: Current primitive parameters (B, N, param_dim)
            union_layer_connections: Union layer weights (B, N)
            branch_assignments: List of branch indices for each primitive
            primitive_control_points: List of initial control points for each primitive
            branches: List of branch point sequences
        """
        B, N, _ = primitive_parameters.shape
        device = primitive_parameters.device
        
        # Initialize total loss
        total_bspline_loss = 0.0
        
        # Get B-spline basis matrix
        bspline_basis = self.bspline_cache.get_bspline_coefficient(
            self.n, self.k, len(self.t_values)
        ).to(device)
        
        # Count primitives per branch for adaptive weighting
        branch_counts = {}
        for assignment in branch_assignments:
            branch_counts[assignment] = branch_counts.get(assignment, 0) + 1
        
        # Calculate branch lengths for weighting
        branch_lengths = {}
        for i, branch in enumerate(branches):
            if len(branch) > 1:
                # Use safe conversion to numpy arrays
                branch_points = []
                for point in branch:
                    branch_points.append(self.safe_to_numpy(point))
                
                # Calculate length using numpy arrays
                length = sum(np.linalg.norm(branch_points[j+1] - branch_points[j]) 
                           for j in range(len(branch_points) - 1))
                branch_lengths[i] = length
            else:
                branch_lengths[i] = 0.1  # Small default for single point branches
        
        # Process each primitive with its assigned branch
        for i in range(N):
            # Get current primitive parameters
            prim_params = primitive_parameters[:, i:i+1, :]
            control_points = prim_params[:, :, :self.n * 3].reshape(-1, self.n, 3)
            
            # Get target branch points
            branch_idx = branch_assignments[i]
            
            # Adaptive weight based on branch assignment and length
            branch_weight = 1.0
            if branch_idx in branch_counts and branch_counts[branch_idx] > 1:
                # Reduce weight for over-assigned branches
                branch_weight = 1.0 / np.sqrt(branch_counts[branch_idx])
            
            if branch_idx in branch_lengths:
                # Weight by branch length (longer branches get more weight)
                total_branch_length = sum(branch_lengths.values())
                if total_branch_length > 0:
                    length_weight = branch_lengths[branch_idx] / total_branch_length
                    branch_weight *= (1.0 + length_weight)
            
            if branch_idx < len(branches) and len(branches[branch_idx]) > 0:
                # Convert branch points to tensor, handling both numpy arrays and existing tensors
                branch_points_list = []
                for point in branches[branch_idx]:
                    if torch.is_tensor(point):
                        # Already a tensor, move to correct device
                        branch_points_list.append(point.to(device).float())
                    else:
                        # Convert from numpy/tuple to tensor using safe conversion
                        point_np = self.safe_to_numpy(point)
                        branch_points_list.append(torch.tensor(point_np, dtype=torch.float32, device=device))
                
                # Stack into a single tensor
                branch_points = torch.stack(branch_points_list).unsqueeze(0)  # Add batch dimension
                
                # Compute B-spline curve from control points
                bspline_curve = torch.matmul(bspline_basis, control_points).reshape(B, -1, 3)
                
                # For over-assigned branches, encourage diversity by adding a small offset
                if branch_idx in branch_counts and branch_counts[branch_idx] > 1:
                    # Add a small diversity factor to avoid identical primitives
                    diversity_offset = 0.01 * (i % branch_counts[branch_idx])  # Different offset for each primitive
                    diversity_noise = torch.randn_like(bspline_curve) * diversity_offset
                    bspline_curve = bspline_curve + diversity_noise
                
                # Compute chamfer distance between B-spline curve and branch points
                primitive_loss = chamfer_distance_3d(bspline_curve, branch_points)
                
                # Apply adaptive weighting
                primitive_loss *= branch_weight
                total_bspline_loss += primitive_loss
            else:
                # No valid branch for this primitive, use light regularization
                control_points_center = control_points.mean(dim=1, keepdim=True)
                regularization_loss = torch.norm(control_points - control_points_center, dim=-1).mean()
                total_bspline_loss += regularization_loss * 0.1  # Light penalty
        
        # Union layer regularization with balance penalty
        union_loss = 0.0
        if union_layer_connections is not None:
            # Standard union layer loss
            union_loss = torch.norm(union_layer_connections, dim=-1).mean()
            
            # Balance penalty: penalize uneven distribution of union weights
            if union_layer_connections.numel() > 0:
                union_weights = union_layer_connections.abs()
                weight_variance = torch.var(union_weights, dim=-1).mean()
                balance_penalty = weight_variance * 0.1  # Encourage balanced weights
                union_loss += balance_penalty
        
        # Return loss dictionary
        return {
            'loss': total_bspline_loss / N + union_loss * 0.1,
            'bspline_loss': total_bspline_loss / N,
            'union_loss': union_loss
        }
        
    def initialize_primitive_parameters(self, primitive_control_points, num_primitives, device):
        """
        Initialize primitive parameters with branch-specific control points.
        
        Args:
            primitive_control_points: List of control points for each primitive
            num_primitives: Number of primitives
            device: Device to put tensors on
            
        Returns:
            Initialized primitive parameters tensor
        """
        B = 1  # Batch size is 1 for single shape training
        param_dim = self.n * 3 + 5  # Control points + 5 additional parameters
        
        # Initialize parameters tensor
        init_params = torch.zeros(B, num_primitives, param_dim).to(device)
        
        # Set control points
        for i in range(num_primitives):
            if i < len(primitive_control_points):
                control_points = torch.as_tensor(primitive_control_points[i], dtype=torch.float32, device=device)
                init_params[0, i, :self.n * 3] = control_points.flatten()
            else:
                # Random initialization for extra primitives
                init_params[0, i, :self.n * 3] = torch.randn(self.n * 3, device=device) * 0.1
        
        # Set default values for additional parameters
        init_params[:, :, self.n * 3:self.n * 3 + 2] = 0.05  # a, b parameters
        init_params[:, :, self.n * 3 + 2] = 2.0  # degree parameter
        init_params[:, :, self.n * 3 + 3:] = 0.0  # scale terms
        
        return init_params

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
