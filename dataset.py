"""Dataset for training and testing ExtrudeNet."""

import os
import open3d as o3d
from torch.utils.data import Dataset
import numpy as np
import config
import h5py
import torch
import branch_utils


class SweepData(Dataset):
    """
    Loading SweepData Dataset with preprocessed voxel, occupancy and skeletal points.
    """

    def __init__(
        self,
        partition="train",
        dataset_root="./data/internet",
        balance=False,
        config=None,
    ):
        super().__init__()
        self.PARTITIONS = ["train", "test", "val"]
        self.dataset_root = dataset_root
        self.partition = partition
        self.balance = balance
        self.num_surface_points = config.num_surface_points
        self.num_testing_points = config.num_testing_points
        self.config = config

        assert (
            self.partition in self.PARTITIONS
        ), "Partition must be either train, test or val.... "

        self.data_urls = [f"{self.dataset_root}/{config.shape_id}"]
        test_names = np.load(self.dataset_root + "/test_names.npz")["names"]
        name_index = np.where(test_names == config.shape_id)

        # Load data
        if self.partition == "test":  # Test with points sampled on the oracle meshes
            with h5py.File(
                os.path.join(self.dataset_root, "ae_voxel_points_samples.hdf5"), "r"
            ) as f:
                points = f["points_64"][:][name_index]
                values = f["values_64"][:][name_index]
                voxels = f["voxels"][:][name_index]
        elif (
            self.partition == "val" or self.partition == "train"
        ):  # Train/Val with points sampled on voxelized meshes
            with h5py.File(os.path.join(self.dataset_root, "voxel2pc.hdf5"), "r") as f:
                data_points = f["points"][:][name_index].astype(float)
                points = data_points[..., :3]
                values = data_points[..., 3][..., None]
                voxels = f["voxels"][:][name_index]

        # Normalize points
        scale = points.max() + 1
        points = (points + 0.5) / scale - 0.5

        testing_points = np.concatenate([points, values], axis=-1)
        self.testing_points = torch.from_numpy(testing_points).float()
        self.voxel = torch.from_numpy(voxels).float().squeeze(-1)
        
        # Load skeletal points and process branch information
        all_skeletal_points = []
        all_branch_assignments = []
        all_primitive_control_points = []
        all_branches = []
        
        for i in range(len(self.data_urls)):
            skeletal_points = np.asarray(
                o3d.io.read_point_cloud(
                    os.path.join(self.data_urls[i], "skeletal_prior.ply")
                ).points
            )
            all_skeletal_points.append(skeletal_points)
            
            # Process branch information
            skeleton_txt_path = os.path.join(self.data_urls[i], "voxel_64_mc.txt")
            if os.path.exists(skeleton_txt_path):
                primitive_control_points, branch_assignments, branches = branch_utils.process_skeleton_for_branch_initialization(
                    skeleton_txt_path, 
                    config.num_primitives, 
                    int(config.bspline_control_points)
                )
                all_branch_assignments.append(branch_assignments)
                all_primitive_control_points.append(primitive_control_points)
                all_branches.append(branches)
            else:
                # Fallback: create default assignments
                all_branch_assignments.append(list(range(config.num_primitives)))
                all_primitive_control_points.append([np.random.randn(int(config.bspline_control_points), 3) * 0.1 for _ in range(config.num_primitives)])
                all_branches.append([])
        
        self.skeletal_points = torch.from_numpy(np.stack(all_skeletal_points)).float()
        self.branch_assignments = all_branch_assignments
        self.primitive_control_points = all_primitive_control_points
        self.branches = all_branches
        self.test_names = list(test_names[name_index])

    def __getitem__(self, item):
        item = 0
        voxel = self.voxel[item]
        testing_points = self.testing_points[item]
        skeletal_points = self.skeletal_points[item]
        branch_assignments = self.branch_assignments[item]
        primitive_control_points = self.primitive_control_points[item]
        branches = self.branches[item]

        # downsample testing point clouds
        if self.balance:
            inner_points = testing_points[testing_points[:, -1] == 1]
            outer_points = testing_points[testing_points[:, -1] == 0]
            inner_index = torch.randint(
                inner_points.shape[0], (self.num_testing_points // 2,)
            )
            outer_index = torch.randint(
                outer_points.shape[0], (self.num_testing_points // 2,)
            )
            testing_points = torch.concatenate(
                [inner_points[inner_index], outer_points[outer_index]], dim=0
            )
        else:
            testing_indices = torch.randint(
                testing_points.shape[0], (self.num_testing_points,)
            )
            testing_points = testing_points[testing_indices]

        skeletal_indices = torch.randint(
            skeletal_points.shape[0],
            (min(skeletal_points.shape[0], self.num_surface_points),),
        )
        skeletal_points = skeletal_points[skeletal_indices]

        return (voxel, testing_points, skeletal_points, self.test_names, branch_assignments, primitive_control_points, branches)

    def __len__(self):
        if self.partition == "train":
            return len(self.data_urls * 200)
        elif self.partition == "val":
            return len(self.data_urls)
        else:
            return 1


class SweepDataPCD(Dataset):
    """
    Loading SweepData Dataset with preprocessed voxel, occupancy and skeletal points.
    """

    def __init__(
        self,
        partition="train",
        dataset_root="./data/internet",
        balance=False,
        config=None,
    ):
        super().__init__()
        self.PARTITIONS = ["train", "test", "val"]
        self.dataset_root = dataset_root
        self.partition = partition
        self.balance = balance
        self.num_surface_points = config.num_surface_points
        self.num_testing_points = config.num_testing_points
        self.config = config

        assert (
            self.partition in self.PARTITIONS
        ), "Partition must be either train, test or val.... "

        self.data_urls = [f"{self.dataset_root}/{config.shape_id}"]
        test_names = np.load(self.dataset_root + "/test_names.npz")["names"]
        name_index = np.where(test_names == config.shape_id)

        # Load data
        if self.partition == "test":  # Test with points sampled on the oracle meshes
            with h5py.File(
                os.path.join(self.dataset_root, "ae_voxel_points_samples.hdf5"), "r"
            ) as f:
                points = f["points_64"][:][name_index]
                values = f["values_64"][:][name_index]
        elif (
            self.partition == "val" or self.partition == "train"
        ):  # Train/Val with points sampled on voxelized meshes
            with h5py.File(os.path.join(self.dataset_root, "voxel2pc.hdf5"), "r") as f:
                data_points = f["points"][:][name_index].astype(float)
                points = data_points[..., :3]
                values = data_points[..., 3][..., None]

        # Normalize points
        scale = points.max() + 1
        points = (points + 0.5) / scale - 0.5
        self.pointcloud = [
            torch.from_numpy(
                np.asarray(
                    o3d.io.read_point_cloud(
                        os.path.join(self.data_urls[0], "model_surface_point_cloud.ply")
                    ).points
                )
            )
        ]

        testing_points = np.concatenate([points, values], axis=-1)
        self.testing_points = torch.from_numpy(testing_points).float()
        
        # Load skeletal points and process branch information
        all_skeletal_points = []
        all_branch_assignments = []
        all_primitive_control_points = []
        all_branches = []
        
        for i in range(len(self.data_urls)):
            # import pdb; pdb.set_trace()
            skeletal_points = np.asarray(
                o3d.io.read_point_cloud(
                    os.path.join(self.data_urls[i], "mcf_skeleton.ply")
                ).points
            )
            all_skeletal_points.append(skeletal_points)
            
            # Process branch information
            skeleton_txt_path = os.path.join(self.data_urls[i], "voxel_64_mc.txt")
            if os.path.exists(skeleton_txt_path):
                primitive_control_points, branch_assignments, branches = branch_utils.process_skeleton_for_branch_initialization(
                    skeleton_txt_path, 
                    config.num_primitives, 
                    int(config.bspline_control_points)
                )
                all_branch_assignments.append(branch_assignments)
                all_primitive_control_points.append(primitive_control_points)
                all_branches.append(branches)
            else:
                # Fallback: create default assignments
                all_branch_assignments.append(list(range(config.num_primitives)))
                all_primitive_control_points.append([np.random.randn(int(config.bspline_control_points), 3) * 0.1 for _ in range(config.num_primitives)])
                all_branches.append([])

        self.skeletal_points = torch.from_numpy(np.stack(all_skeletal_points)).float()
        self.branch_assignments = all_branch_assignments
        self.primitive_control_points = all_primitive_control_points
        self.branches = all_branches
        self.test_names = list(test_names[name_index])

    def __getitem__(self, item):
        # Overfitting to one instance
        item = 0
        pointcloud = self.pointcloud[0].float()
        testing_points = self.testing_points[item]
        skeletal_points = self.skeletal_points[item]
        branch_assignments = self.branch_assignments[item]
        primitive_control_points = self.primitive_control_points[item]
        branches = self.branches[item]

        # downsample testing point clouds
        if self.balance:
            inner_points = testing_points[testing_points[:, -1] == 1]
            outer_points = testing_points[testing_points[:, -1] == 0]
            inner_index = torch.randint(
                inner_points.shape[0], (self.num_testing_points // 2,)
            )
            outer_index = torch.randint(
                outer_points.shape[0], (self.num_testing_points // 2,)
            )
            testing_points = torch.concatenate(
                [inner_points[inner_index], outer_points[outer_index]], dim=0
            )
        else:
            testing_indices = torch.randint(
                testing_points.shape[0], (self.num_testing_points,)
            )
            testing_points = testing_points[testing_indices]

        skeletal_indices = torch.randint(
            skeletal_points.shape[0],
            (min(skeletal_points.shape[0], self.num_surface_points),),
        )
        skeletal_points = skeletal_points[skeletal_indices]

        surface_indices = np.random.randint(
            0, pointcloud.shape[0], self.num_surface_points
        )
        pointcloud = pointcloud[surface_indices]

        return (
            pointcloud.unsqueeze(0),
            testing_points,
            skeletal_points,
            self.test_names,
            branch_assignments,
            primitive_control_points,
            branches,
        )

    def __len__(self):
        if self.partition == "train":
            return len(self.data_urls * 200)
        elif self.partition == "val":
            return len(self.data_urls)
        else:
            return 1
