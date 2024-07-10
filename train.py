import os
import shutil
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import SweepData, SweepDataPCD
from config import Config
import utils
import gc

from trainer import trainer
import lightning as L
from lightning.pytorch.tuner import Tuner
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

torch.set_float32_matmul_precision("high")
torch.manual_seed(42)
device = torch.device("cuda")


def train(train_config):
    """Training script."""
    # Initialise training configuration
    utils.init(train_config)

    # Data loader
    train_loader = DataLoader(
        globals()[train_config.dataset](
            dataset_root=train_config.dataset_root,
            balance=False,
            partition="train",
            config=train_config,
        ),
        num_workers=31,
        batch_size=train_config.train_batch_size_per_gpu * train_config.num_gpu,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        globals()[train_config.dataset](
            dataset_root=train_config.dataset_root,
            balance=False,
            partition="val",
            config=train_config,
        ),
        num_workers=31,
        batch_size=train_config.test_batch_size_per_gpu * train_config.num_gpu,
        shuffle=False,
        drop_last=True,
    )

    test_loader = DataLoader(
        globals()[train_config.dataset](
            dataset_root=train_config.dataset_root,
            balance=False,
            partition="test",
            config=train_config,
        ),
        num_workers=31,
        batch_size=train_config.test_batch_size_per_gpu * train_config.num_gpu,
        shuffle=False,
        drop_last=True,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=f"{train_config.sample_dir}/{train_config.experiment_name}",
        filename="best",
        save_top_k=1,
        mode="min",
        every_n_epochs=train_config.save_every_epoch,
    )

    # Initialize the model for a warm start of primitives
    init_model = trainer.InitializationTrainer(train_config, pcd=train_config.pcd).to(
        device
    )
    early_stop_callback = EarlyStopping(
        monitor="loss_total", stopping_threshold=0.12, patience=10
    )
    init_trainer = L.Trainer(
        max_epochs=10000,
        default_root_dir=f"./{train_config.sample_dir}/{train_config.experiment_name}",
        callbacks=[early_stop_callback],
    )
    init_trainer.fit(init_model, train_loader)
    init_trainer.save_checkpoint(
        f"{train_config.sample_dir}/{train_config.experiment_name}/init.ckpt"
    )
    init_trainer.test(
        init_model,
        test_loader,
        ckpt_path=f"{train_config.sample_dir}/{train_config.experiment_name}/init.ckpt",
    )

    # Train SweepNet
    model = trainer.Trainer(
        train_config,
        pcd=train_config.pcd,
        checkpoint_path=f"{train_config.sample_dir}/{train_config.experiment_name}/init.ckpt",
    ).to(device)
    # Training for a longer epoch will have better results, 10 epoch is satisfactory for a quick test
    sweepnet_trainer = L.Trainer(
        max_epochs=train_config.epoch,
        default_root_dir=f"./{train_config.sample_dir}/{train_config.experiment_name}",
        callbacks=[checkpoint_callback],
    )
    # Train
    sweepnet_trainer.fit(model, train_loader, val_loader)

    # Test
    sweepnet_trainer.test(
        model,
        test_loader,
        ckpt_path=f"{train_config.sample_dir}/{train_config.experiment_name}/best.ckpt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SweepNet Training Script.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/default.json",
        metavar="N",
        help="config_path",
    )

    args = parser.parse_args()
    config = Config(args.config_path)

    # Backup other code scripts
    src_dir = os.path.dirname(os.path.abspath(__file__))
    dst_dir = os.path.join(src_dir, config.sample_dir, config.experiment_name, "code")
    os.makedirs(dst_dir, exist_ok=True)
    for file in os.listdir(src_dir):
        if file.endswith(".py"):
            shutil.copy2(os.path.join(src_dir, file), dst_dir)
    shutil.copy2(args.config_path, dst_dir)

    # Train
    train(config)

    # Clean up
    gc.collect()
