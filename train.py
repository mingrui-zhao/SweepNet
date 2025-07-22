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
    
    # Configure initialization based on mode
    if train_config.use_numerical_init:
        print("=== Using Numerical Initialization Mode ===")
        print("Training network to overfit to numerical parameters for fast initialization...")
        # Fast overfitting for numerical initialization (500 epochs max)
        init_epochs = min(train_config.numerical_init_epochs, 500)
        early_stop_callback = EarlyStopping(
            monitor="loss_total", 
            stopping_threshold=0.005,  # Very low threshold for fast convergence
            patience=15,  # Shorter patience for faster stopping
            min_delta=0.001
        )
    else:
        print("=== Using Branch-wise Initialization Mode ===")
        print("Training with branch-wise loss for detailed initialization...")
        # More epochs for learning-based initialization
        init_epochs = 1000
        early_stop_callback = EarlyStopping(
            monitor="loss_total", 
            stopping_threshold=0.1, 
            patience=25,
            min_delta=0.01
        )
    
    init_trainer = L.Trainer(
        max_epochs=init_epochs,
        default_root_dir=f"./{train_config.sample_dir}/{train_config.experiment_name}",
        callbacks=[early_stop_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
    print(f"Starting initialization training for up to {init_epochs} epochs...")
    print(f"Learning rate: {init_model.learning_rate:.6f}")
    print(f"Early stopping patience: {early_stop_callback.patience}")
    
    init_trainer.fit(init_model, train_loader)
    
    # Save the initialization checkpoint manually (in case on_training_end wasn't called)
    init_checkpoint_path = f"{train_config.sample_dir}/{train_config.experiment_name}/init_modules.pth"
    if not os.path.exists(init_checkpoint_path):
        print("Saving initialization checkpoint manually...")
        torch.save({
            "encoder_state_dict": init_model.model.encoder.state_dict(),
            "decoder_state_dict": init_model.model.decoder.state_dict(),
            "selection_head_state_dict": init_model.model.selection_head.state_dict(),
            "swept_volume_head_state_dict": init_model.model.swept_volume_head.state_dict(),
            "config": train_config,
            "use_numerical_init": init_model.use_numerical_init,
            "use_branch_wise_init": init_model.use_branch_wise_init
        }, init_checkpoint_path)
        print(f"Initialization checkpoint saved to: {init_checkpoint_path}")
    else:
        print(f"Initialization checkpoint already exists at: {init_checkpoint_path}")
    
    # Test initialization
    init_trainer.test(init_model, test_loader)
    
    print("Initialization complete. Starting main training with global skeleton supervision...")
    
    # Check if initialization checkpoint exists
    if not os.path.exists(init_checkpoint_path):
        print(f"Warning: Initialization checkpoint not found at {init_checkpoint_path}")
        print("Starting main training without initialization checkpoint...")
        checkpoint_path = None
    else:
        print(f"Using initialization checkpoint: {init_checkpoint_path}")
        checkpoint_path = init_checkpoint_path
    
    # Train SweepNet (main training uses global skeleton supervision)
    # Use the efficient module-only checkpoint loading
    model = trainer.Trainer(
        train_config,
        pcd=train_config.pcd,
        checkpoint_path=checkpoint_path,  # Use the efficient module-only checkpoint
    ).to(device)
    
    # Training for a longer epoch will have better results, 10 epoch is satisfactory for a quick test
    sweepnet_trainer = L.Trainer(
        max_epochs=train_config.epoch,
        default_root_dir=f"./{train_config.sample_dir}/{train_config.experiment_name}",
        callbacks=[checkpoint_callback],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        log_every_n_steps=10,
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
