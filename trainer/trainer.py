import os
import torch
from torch import nn
import lightning as L
from model import SweepNet, SweepNetPCD
from loss import Loss, InitLoss
from neural_sweeper import poco_model
import utils

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
        self.learning_rate = config.learning_rate
        self.scale_bspline_loss = config.scale_bspline_loss
        self.criterion = InitLoss(self.config)

    def training_step(self, batch, batch_idx):
        voxel, points, skeletal_points, _ = batch
        feature = self.model.encoder(voxel)
        code = self.model.decoder(feature)
        union_layer_connections = self.model.selection_head(code, is_training=True)
        primitive_parameters = self.model.swept_volume_head(code)
        mask = (union_layer_connections > 0.0).flatten()
        primitive_parameters_selected = primitive_parameters[:, mask, :]

        loss = self.criterion(
            skeletal_points, primitive_parameters, union_layer_connections
        )
        self.log_dict(loss, on_step=True, prog_bar=True, logger=True, batch_size=1)
        return loss["loss_total"]

    def test_step(self, batch, batch_idx):
        voxel, points, skeletal_points, _ = batch
        feature = self.model.encoder(voxel)
        code = self.model.decoder(feature)
        primitive_parameters = self.model.swept_volume_head(code)
        out_path = (
            f"./{self.config.sample_dir}/{self.config.experiment_name}/init_axis.ply"
        )
        utils.render_general_cyl(
            primitive_parameters[:, :, :9].detach().cpu().numpy(),
            out_path,
            tube_save=False,
        )
        utils.vis_mesh(self.model, voxel, 64, 16, "initialization", self.config)

    # def on_training_end(self):
    #     # save model state dict of used heads in one file
    #     torch.save({
    #         "encoder_state_dict": self.model.encoder.state_dict(),
    #         "decoder_state_dict": self.model.decoder.state_dict(),
    #         "selection_head_state_dict": self.model.selection_head.state_dict(),
    #         "swept_volume_head_state_dict": self.model.swept_volume_head.state_dict()
    #     }, os.path.join(self.default_root_dir,"init.pth"))

    def configure_optimizers(self):
        params_to_optimize = [
            {"params": self.model.encoder.parameters()},
            {"params": self.model.decoder.parameters()},
            {"params": self.model.selection_head.parameters()},
            {"params": self.model.swept_volume_head.parameters()},
        ]
        optimizer = torch.optim.Adam(
            params_to_optimize, lr=self.learning_rate, betas=(self.config.beta1, 0.999)
        )
        return optimizer


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
        # Load the state dict into the model
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint["state_dict"])

    def training_step(self, batch, batch_idx):
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
        return loss["loss_total"]

    def validation_step(self, batch, batch_idx):
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
        utils.vis_mesh(self.model, voxel, 64, 16, self.current_epoch, self.config)

    def test_step(self, batch, batch_idx):
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
        utils.generate_mesh_primitives(
            self.model,
            voxel,
            self.config,
            bspline_cache=self.bspline_cache,
            n=self.config.bspline_control_points,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.config.beta1, 0.999),
        )
        return optimizer
