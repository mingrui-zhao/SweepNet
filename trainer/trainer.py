import os
import torch
from torch import nn
import lightning as L
from model import SweepNet, SweepNetPCD
from loss import Loss, InitLoss, BranchWiseInitLoss
from neural_sweeper import poco_model
import utils
import numpy as np

torch.manual_seed(42)


class InitializationTrainer(L.LightningModule):
    def __init__(self, config, pcd=False):
        super(InitializationTrainer, self).__init__()
        # load neural sweeper
        neural_sweeper = poco_model.get_poco_model("./neural_sweeper/poco_config.yaml")
        ns_ckpt = torch.load(config.neural_sweeper_path)
        neural_sweeper.load_state_dict(ns_ckpt["model_state_dict"])
        neural_sweeper.eval()

        # Freeze neural sweeper weights
        for param in neural_sweeper.parameters():
            param.requires_grad = False

        # loading bspline cache
        self.bspline_cache = utils.BsplineCache()

        # loading model
        if pcd:
            model = SweepNetPCD(
                config, neural_sweeper, bspline_cache=self.bspline_cache
            )
        else:
            model = SweepNet(config, neural_sweeper, bspline_cache=self.bspline_cache)
        self.model = model
        self.config = config
        
        # Optimized learning rates for faster convergence
        self.learning_rate = config.learning_rate * 2.0  # Double the learning rate for faster convergence
        self.scale_bspline_loss = config.scale_bspline_loss
        self.criterion = BranchWiseInitLoss(self.config)
        
        # Initialization mode flags
        self.use_numerical_init = getattr(config, 'use_numerical_init', True)
        self.use_branch_wise_init = getattr(config, 'use_branch_wise_init', True)
        self.numerical_init_epochs = getattr(config, 'numerical_init_epochs', 500)  # Fast overfitting for numerical init
        
        # Save neural sweeper separately for efficiency
        self.neural_sweeper = neural_sweeper

    def numerical_branch_initialization(self, branch_assignments, primitive_control_points, branches):
        """
        Directly set primitive parameters based on branch geometry using PyTorch operations.
        This is much faster than the learning-based approach and uses numerical computation.
        """
        device = next(self.model.parameters()).device
        B, N = 1, self.config.num_primitives
        param_dim = self.config.bspline_control_points * 3 + 5
        
        # Initialize parameters tensor
        init_params = torch.zeros(B, N, param_dim, device=device, dtype=torch.float32)
        
        # Set control points directly from branch geometry using PyTorch operations
        for i in range(N):
            if i < len(primitive_control_points):
                control_points = torch.as_tensor(primitive_control_points[i], dtype=torch.float32, device=device)
                init_params[0, i, :self.config.bspline_control_points * 3] = control_points.flatten()
            else:
                # Random initialization for extra primitives
                init_params[0, i, :self.config.bspline_control_points * 3] = torch.randn(
                    self.config.bspline_control_points * 3, device=device, dtype=torch.float32
                ) * 0.1
        
        # Set profile parameters based on branch characteristics using PyTorch operations
        for i in range(N):
            branch_idx = branch_assignments[i]
            if branch_idx < len(branches) and len(branches[branch_idx]) > 1:
                # Convert branch points to PyTorch tensors for computation
                branch_points = []
                for point in branches[branch_idx]:
                    if torch.is_tensor(point):
                        branch_points.append(point.to(device).float())
                    else:
                        branch_points.append(torch.tensor(point, dtype=torch.float32, device=device))
                
                # Calculate branch length using PyTorch operations
                branch_tensor = torch.stack(branch_points)
                if len(branch_tensor) > 1:
                    # Compute differences between consecutive points
                    diffs = branch_tensor[1:] - branch_tensor[:-1]
                    # Compute segment lengths
                    segment_lengths = torch.norm(diffs, dim=1)
                    # Total branch length
                    branch_length = torch.sum(segment_lengths).item()
                else:
                    branch_length = 0.1
                
                # Set profile parameters based on branch length
                # Longer branches get slightly larger cross-sections
                base_radius = 0.02 + min(0.03, branch_length * 0.01)
                init_params[0, i, self.config.bspline_control_points * 3:self.config.bspline_control_points * 3 + 2] = base_radius
                init_params[0, i, self.config.bspline_control_points * 3 + 2] = 2.0  # degree parameter
            else:
                # Default values for primitives without valid branches
                init_params[0, i, self.config.bspline_control_points * 3:self.config.bspline_control_points * 3 + 2] = 0.05
                init_params[0, i, self.config.bspline_control_points * 3 + 2] = 2.0
        
        # Set scale terms to zero
        init_params[:, :, self.config.bspline_control_points * 3 + 3:] = 0.0
        
        return init_params

    def training_step(self, batch, batch_idx):
        if self.use_branch_wise_init and len(batch) >= 7:
            # Branch-wise initialization (uses branch-wise skeleton supervision)
            voxel, points, skeletal_points, _, branch_assignments, primitive_control_points, branches = batch
            
            if self.use_numerical_init:
                # Mode 1: Numerical initialization - train network to overfit to numerical parameters
                numerical_params = self.numerical_branch_initialization(
                    branch_assignments, primitive_control_points, branches
                )
                
                # Forward pass
                feature = self.model.encoder(voxel)
                code = self.model.decoder(feature)
                union_layer_connections = self.model.selection_head(code, is_training=True)
                primitive_parameters = self.model.swept_volume_head(code)
                
                # Loss to match numerical initialization (fast overfitting)
                loss_params = torch.nn.MSELoss()(primitive_parameters, numerical_params)
                
                # Union layer loss - encourage all primitives to be active initially
                dummy_union = torch.ones_like(union_layer_connections)
                loss_weight = torch.nn.MSELoss()(union_layer_connections, dummy_union)
                
                # Combined loss with emphasis on parameter matching for fast convergence
                loss_total = loss_params + 0.1 * loss_weight
                
                loss_dict = {
                    "loss_bspline": loss_params,
                    "loss_param": loss_params,
                    "loss_weight": loss_weight,
                    "loss_total": loss_total,
                }
            else:
                # Mode 2: Branch-wise initialization with branch-wise loss
                feature = self.model.encoder(voxel)
                code = self.model.decoder(feature)
                union_layer_connections = self.model.selection_head(code, is_training=True)
                primitive_parameters = self.model.swept_volume_head(code)
                
                # Initialize with branch-specific control points if this is the first few steps
                if batch_idx < 10:  # Initialize for first 10 steps
                    device = voxel.device
                    initial_params = self.criterion.initialize_primitive_parameters(
                        primitive_control_points, self.config.num_primitives, device
                    )
                    
                    # Blend the initial parameters with the model's prediction
                    alpha = max(0.0, 1.0 - batch_idx / 10.0)  # Reduce initialization influence over time
                    primitive_parameters = alpha * initial_params + (1 - alpha) * primitive_parameters
                
                # Compute branch-wise loss
                loss_dict = self.criterion(
                    skeletal_points, primitive_parameters, union_layer_connections,
                    branch_assignments, primitive_control_points, branches
                )
                
        else:
            # Fallback to original global skeleton initialization
            voxel, points, skeletal_points, _ = batch[:4]
            feature = self.model.encoder(voxel)
            code = self.model.decoder(feature)
            union_layer_connections = self.model.selection_head(code, is_training=True)
            primitive_parameters = self.model.swept_volume_head(code)
            
            # Use original InitLoss (global skeleton supervision)
            original_criterion = InitLoss(self.config)
            loss_dict = original_criterion(
                skeletal_points, primitive_parameters, union_layer_connections
            )
        
        self.log_dict(loss_dict, on_step=True, prog_bar=True, logger=True, batch_size=1)
        
        # Handle different loss key names
        if "loss_total" in loss_dict:
            return loss_dict["loss_total"]
        elif "loss" in loss_dict:
            return loss_dict["loss"]
        else:
            # Fallback: return first loss value
            return next(iter(loss_dict.values()))

    def test_step(self, batch, batch_idx):
        if len(batch) >= 7:
            voxel, points, skeletal_points, _, branch_assignments, primitive_control_points, branches = batch
        else:
            voxel, points, skeletal_points, _ = batch[:4]
            
        feature = self.model.encoder(voxel)
        code = self.model.decoder(feature)
        primitive_parameters = self.model.swept_volume_head(code)
        
        # Save B-spline axis visualization
        out_path = (
            f"./{self.config.sample_dir}/{self.config.experiment_name}/init_axis.ply"
        )
        utils.render_general_cyl(
            primitive_parameters[:, :, :9].detach().cpu().numpy(),
            out_path,
            tube_save=False,
        )
        
        # Generate the reconstructed mesh (original functionality)
        utils.vis_mesh(self.model, voxel, 64, 16, "initialization", self.config)

    def on_training_end(self):
        """Save only the trainable modules (excluding neural sweeper) for efficiency."""
        save_path = os.path.join(self.default_root_dir, "init_modules.pth")
        torch.save({
            "encoder_state_dict": self.model.encoder.state_dict(),
            "decoder_state_dict": self.model.decoder.state_dict(),
            "selection_head_state_dict": self.model.selection_head.state_dict(),
            "swept_volume_head_state_dict": self.model.swept_volume_head.state_dict(),
            "config": self.config,
            "use_numerical_init": self.use_numerical_init,
            "use_branch_wise_init": self.use_branch_wise_init
        }, save_path)
        print(f"Saved trainable modules to {save_path}")

    def configure_optimizers(self):
        # Optimized learning rate for faster convergence
        params_to_optimize = [
            {"params": self.model.encoder.parameters(), "lr": self.learning_rate},
            {"params": self.model.decoder.parameters(), "lr": self.learning_rate},
            {"params": self.model.selection_head.parameters(), "lr": self.learning_rate * 1.5},  # Higher LR for selection
            {"params": self.model.swept_volume_head.parameters(), "lr": self.learning_rate * 1.2},  # Higher LR for parameters
        ]
        optimizer = torch.optim.Adam(
            params_to_optimize, lr=self.learning_rate, betas=(self.config.beta1, 0.999)
        )
        
        # Add learning rate scheduler for faster convergence
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss_total",
            },
        }


class Trainer(L.LightningModule):
    def __init__(self, config, checkpoint_path=None, pcd=False):
        super(Trainer, self).__init__()
        self.config = config
        # load neural sweeper
        neural_sweeper = poco_model.get_poco_model("./neural_sweeper/poco_config.yaml")
        ns_ckpt = torch.load(config.neural_sweeper_path)
        neural_sweeper.load_state_dict(ns_ckpt["model_state_dict"])
        neural_sweeper.eval()

        # Freeze neural sweeper weights
        for param in neural_sweeper.parameters():
            param.requires_grad = False

        # loading bspline cache
        self.bspline_cache = utils.BsplineCache()
        # loading model
        if pcd:
            model = SweepNetPCD(
                config, neural_sweeper, bspline_cache=self.bspline_cache
            )
        else:
            model = SweepNet(config, neural_sweeper, bspline_cache=self.bspline_cache)
        self.model = model

        self.learning_rate = config.learning_rate
        self.scale_bspline_loss = config.scale_bspline_loss
        self.scale_recon_loss = config.scale_recon_loss
        self.scale_overlap_loss = config.scale_overlap_loss
        self.scale_parsimony_loss = config.scale_parsimony_loss
        self.criterion = Loss(config)
        self.checkpoint_path = checkpoint_path
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint with support for both full model and module-only saves."""
        if checkpoint_path is None:
            print("No checkpoint provided, starting with random initialization...")
            return
            
        checkpoint = torch.load(checkpoint_path)
        
        if "encoder_state_dict" in checkpoint:
            # Load module-only checkpoint (efficient loading)
            print("Loading module-only checkpoint...")
            self.model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            self.model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
            self.model.selection_head.load_state_dict(checkpoint["selection_head_state_dict"])
            self.model.swept_volume_head.load_state_dict(checkpoint["swept_volume_head_state_dict"])
        else:
            # Load full model checkpoint (backward compatibility)
            print("Loading full model checkpoint...")
            self.load_state_dict(checkpoint["state_dict"])

    def training_step(self, batch, batch_idx):
        # Handle both old format (4 values) and new format (7 values) with branch info
        if len(batch) >= 7:
            voxel, points, skeletal_points, _, branch_assignments, primitive_control_points, branches = batch
        else:
            voxel, points, skeletal_points, _ = batch
            
        occupancies, occupancies_pre_union, primitive_parameters, selection_matrix = (
            self.model(voxel, points[:, :, :3], is_training=True)
        )
        predict_occupancies = (occupancies >= 0.5).float()
        target_occupancies = (points[:, :, -1]).float()
        accuracy = torch.sum(predict_occupancies * target_occupancies) / torch.sum(
            target_occupancies
        )
        recall = torch.sum(predict_occupancies * target_occupancies) / (
            torch.sum(predict_occupancies) + 1e-9
        )

        loss = self.criterion(
            occupancies,
            target_occupancies,
            occupancies_pre_union,
            primitive_parameters,
            selection_matrix,
            skeletal_points,
        )
        loss["accuracy"] = accuracy
        loss["recall"] = recall
        self.log_dict(loss, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        
        # Handle different loss key names
        if "loss_total" in loss:
            return loss["loss_total"]
        elif "loss" in loss:
            return loss["loss"]
        else:
            # Fallback: return first loss value
            return next(iter(loss.values()))

    def validation_step(self, batch, batch_idx):
        # Handle both old format (4 values) and new format (7 values) with branch info
        if len(batch) >= 7:
            voxel, points, skeletal_points, _, branch_assignments, primitive_control_points, branches = batch
        else:
            voxel, points, skeletal_points, _ = batch
            
        occupancies, occupancies_pre_union, primitive_parameters, selection_matrix = (
            self.model(voxel, points[:, :, :3], is_training=False)
        )
        predict_occupancies = (occupancies >= 0.5).float()
        target_occupancies = (points[:, :, -1]).float()
        accuracy = torch.sum(predict_occupancies * target_occupancies) / torch.sum(
            target_occupancies
        )
        recall = torch.sum(predict_occupancies * target_occupancies) / (
            torch.sum(predict_occupancies) + 1e-9
        )

        loss = self.criterion(
            occupancies,
            target_occupancies,
            occupancies_pre_union,
            primitive_parameters,
            selection_matrix,
            skeletal_points,
        )
        loss["accuracy"] = accuracy
        loss["recall"] = recall
        loss["val_loss"] = loss["loss_recon"]
        self.log_dict(loss, on_step=True, prog_bar=True, logger=True, batch_size=1)
        
        # Generate visualizations every few epochs to avoid too many files
        if self.current_epoch % self.config.vis_interval == 0:
            # Generate the reconstructed mesh (original functionality)
            utils.vis_mesh(self.model, voxel, 64, 16, self.current_epoch, self.config)

    def test_step(self, batch, batch_idx):
        # Handle both old format (4 values) and new format (7 values) with branch info
        if len(batch) >= 7:
            voxel, points, skeletal_points, _, branch_assignments, primitive_control_points, branches = batch
        else:
            voxel, points, skeletal_points, _ = batch
            
        occupancies, occupancies_pre_union, primitive_parameters, selection_matrix = (
            self.model(voxel, points[:, :, :3], is_training=False)
        )
        predict_occupancies = (occupancies >= 0.5).float()
        target_occupancies = (points[:, :, -1]).float()
        accuracy = torch.sum(predict_occupancies * target_occupancies) / torch.sum(
            target_occupancies
        )
        recall = torch.sum(predict_occupancies * target_occupancies) / (
            torch.sum(predict_occupancies) + 1e-9
        )

        loss = self.criterion(
            occupancies,
            target_occupancies,
            occupancies_pre_union,
            primitive_parameters,
            selection_matrix,
            skeletal_points,
        )
        loss["accuracy"] = accuracy
        loss["recall"] = recall
        self.log_dict(loss, on_epoch=True, prog_bar=True, logger=True, batch_size=1)
        
        # Generate final visualizations for test step
        print("Generating final ground truth swept volumes...")
        utils.generate_mesh_primitives(self.model, voxel, self.config)
        # utils.create_gt_swept_volume(
        #     primitive_parameters.detach().cpu().numpy(),
        #     num_control_points=self.config.bspline_control_points,
        #     file_prefix=f"./{self.config.sample_dir}/{self.config.experiment_name}",
        #     edit_suffix="final"
        # )
        print("Ground truth swept volumes generated successfully!")
            
        
        # Handle different loss key names
        if "loss_total" in loss:
            return loss["loss_total"]
        elif "loss" in loss:
            return loss["loss"]
        else:
            # Fallback: return first loss value
            return next(iter(loss.values()))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.config.beta1, 0.999),
        )
        return optimizer
