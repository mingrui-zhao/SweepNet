"""Configurations for training and testing SweepNet"""

import json
import os
import argparse


class Config:
    """Configuration class."""

    def __init__(self, config_path):
        with open(config_path, "r", encoding="utf-8") as json_file:  # ignore
            config_dict = json.load(json_file)

        # Neural Sweeper
        neural_sweeper_dir = "./neural_sweeper/ckpt"
        self.neural_sweeper_path = os.path.join(
            neural_sweeper_dir,
            "neural_sweeper.pth",
        )

        # Dataset
        self.dataset = config_dict["dataset"]
        self.dataset_root = config_dict["dataset_root"]
        self.num_surface_points = config_dict["num_surface_points"]
        self.num_sample_points = config_dict["num_sample_points"]
        self.num_testing_points = config_dict["num_testing_points"]
        self.balance = config_dict["balance"]

        # Experiment related
        self.pcd = config_dict["pcd"]
        self.shape_id = config_dict["shape_id"]
        experiment_prefix = f"{self.shape_id}_"
        experiment_identifier = config_dict["experiment_identifier"]
        self.experiment_name = experiment_prefix + experiment_identifier
        self.save_every_epoch = config_dict["save_every_epoch"]
        self.vis_interval = config_dict["vis_interval"]
        self.vis_res = config_dict["vis_res"]

        # Eval
        self.real_size = config_dict["real_size"]
        self.test_size = config_dict["test_size"]
        self.sample_dir = config_dict["sample_dir"]

        # Hardware related
        self.num_gpu = config_dict["num_gpu"]
        self.train_batch_size_per_gpu = config_dict["train_batch_size_per_gpu"]
        self.test_batch_size_per_gpu = config_dict["test_batch_size_per_gpu"]

        # Loss
        self.scale_bspline_loss = config_dict["scale_bspline_loss"]
        self.scale_recon_loss = config_dict["scale_recon_loss"]
        self.scale_overlap_loss = config_dict["scale_overlap_loss"]
        self.overlap_threshold = config_dict["overlap_threshold"]
        self.scale_parsimony_loss = config_dict["scale_parsimony_loss"]

        # Optimizer
        self.learning_rate = config_dict["learning_rate"]
        self.beta1 = config_dict["beta1"]

        # Sweep surfaces
        self.bspline_control_points = int(config_dict["bspline_control_points"])
        self.bspline_order = config_dict["bspline_order"]
        self.num_primitives = int(config_dict["num_primitives"])
        self.feature_dim = config_dict["feature_dim"]
        self.sharpness = config_dict["sharpness"]

        # Training
        self.epoch = config_dict["epoch"]
        self.eval_interval = config_dict["eval_interval"]
        self.drop_interval = config_dict["drop_interval"]
