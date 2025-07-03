# File:         infer.py
# Date:         2024/06/03
# Description:  Top-level script to infer a pre-trained model on a specific dataset

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from yaml import safe_load

from fignet.data_loader import MujocoDataset, MoviDataset, ToTensor, collate_fn
from fignet.scene import Scene
from fignet.simulator import LearnedSimulator
from fignet.types import KinematicType, NodeType
from fignet.utils import rollout, to_numpy


MAX_ROLLOUT_DURATION: int = 105


@torch.no_grad()
def compute_rollouts(
    config: dict, model: torch.nn.Module, device: torch.device
) -> None:
    if config["dataset_type"] == "Mujoco":
        dataset_builder = MujocoDataset
    elif config["dataset_type"] == "Movi":
        dataset_builder = MoviDataset
    else:
        raise RuntimeError(f"ERROR: Unknown dataset type {config['dataset_type']}")

    # Instanciate the dataset
    dataset = dataset_builder(
        config["test_data_path"],
        split="test",
        mode="trajectory",
        input_sequence_length=MAX_ROLLOUT_DURATION,
        transform=T.Compose([ToTensor(device)]),
        config=config.get("data_config"),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    checkpoint_name = Path(config["checkpoint"]).stem
    os.makedirs(
        os.path.join(config["logging_folder"], checkpoint_name, "rollout"),
        exist_ok=True,
    )

    model.eval()
    for data in tqdm(data_loader, desc="Rollout trajectories", total=len(data_loader)):
        scene_config = data[2]
        output_file = os.path.join(
            config["logging_folder"],
            checkpoint_name,
            "rollout",
            f"{scene_config['experiment_id']}.npy",
        )
        if os.path.exists(output_file):
            continue

        traj = data[0]
        init_poses = traj["pose_seq"][
            0 : config["data_config"]["input_seq_length"], ...
        ]
        gt_poses = to_numpy(traj["pose_seq"])
        gt_poses = np.delete(gt_poses, 0, axis=0)
        obj_ids = traj["obj_ids"]
        for k, v in obj_ids.items():
            obj_ids[k] = v.cpu().item()
        pred_traj = rollout(
            sim=model,
            init_obj_poses=init_poses,
            obj_ids=obj_ids,
            scene=Scene(scene_config),
            device=device,
            nsteps=MAX_ROLLOUT_DURATION,
            quiet=True,
        )
        pred_traj = np.concatenate([init_poses.cpu(), pred_traj], axis=0)
        pred_traj = np.expand_dims(pred_traj, axis=0)  # [1, duration, objects, 3 + 4]
        np.save(
            os.path.join(
                config["logging_folder"],
                checkpoint_name,
                "rollout",
                f"{scene_config['experiment_id']}.npy",
            ),
            pred_traj,
        )


@torch.no_grad()
def compute_onestep_errors(
    config: dict,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    """Compute one-step prediction errors on the complete test set

    Returns
    -------
    np.ndarray[batches x 2]
        MSE (col 0) and MAE (col 1) per batch (row)
    """
    if config["dataset_type"] == "Mujoco":
        dataset_builder = MujocoDataset
    elif config["dataset_type"] == "Movi":
        dataset_builder = MoviDataset
    else:
        raise RuntimeError(f"ERROR: Unknown dataset type {config['dataset_type']}")

    # Instanciate the dataset
    dataset = dataset_builder(
        config["test_data_path"],
        split="test",
        mode="sample",
        input_sequence_length=config["data_config"]["input_seq_length"],
        transform=T.Compose([ToTensor(device)]),
        config=config.get("data_config"),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        pin_memory=False,
        collate_fn=collate_fn,
    )

    # Compute the one-step errors from the model
    onestep_mse: list[float] = []
    onestep_mae: list[float] = []
    model.eval()
    for sample in tqdm(data_loader, desc="Onestep errors", total=len(data_loader)):
        non_kinematic_mask = (
            sample.node_sets[NodeType.MESH].kinematic == KinematicType.DYNAMIC
        ).to(device)
        sample = ToTensor(device)(sample)
        pred_acc, _ = model.predict_accelerations(sample)
        target_acc = model.normalize_accelerations(
            sample.node_sets[NodeType.MESH].target
        )

        # Compute Mean Square Error loss
        num_non_kinematic = non_kinematic_mask.sum()
        mse = torch.nn.functional.mse_loss(pred_acc, target_acc, reduction="none")
        mse = mse.sum(dim=-1)
        mse = mse * non_kinematic_mask.squeeze()
        mse = mse.sum() / num_non_kinematic
        onestep_mse.append(mse.item())

        # Compute Mean Absolute Error
        mae = torch.nn.functional.l1_loss(pred_acc, target_acc, reduction="none")
        mae = mae.sum(dim=-1)
        mae = mae * non_kinematic_mask.squeeze()
        mae = mae.sum() / num_non_kinematic
        onestep_mae.append(mae.item())

    return np.stack([onestep_mse, onestep_mae], axis=1)


def model_from_checkpoint(config: dict, device: torch.device) -> torch.nn.Module:
    # Instanciate the model from the checkpoint
    model = LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=config["latent_dim"],
        nmessage_passing_steps=config["message_passing_steps"],
        nmlp_layers=config["mlp_layers"],
        mlp_hidden_dim=config["latent_dim"],
        leave_out_mm=config["leave_out_mm"],
        device=device.type,
    )
    model.load(config["checkpoint"])
    model = model.to(device)

    return model


def main(args: argparse.Namespace) -> int:
    # Make sure the config file exists
    config_path = os.path.join(os.getcwd(), args.config_file)
    if not os.path.isfile(config_path):
        print(f"ERROR: Cannot find config file {config_path}")
        return 1

    # Make sure the checkpoints exists
    checkpoint_path = os.path.join(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        print(f"ERROR: Cannot find modle checkpoint {checkpoint_path}")
        return 1

    # Parse the config file
    with open(config_path) as f:
        config = safe_load(f)
    config["checkpoint"] = checkpoint_path

    # Override the logging_folder if requested
    if args.logging_folder is not None and args.logging_folder != "":
        config["logging_folder"] = args.logging_folder
        print(f"WARN: Overriding logging folder: {args.logging_folder}")

    # Adjust some inference parameters
    if torch.cuda.is_available() and config.get("use_cuda", True):
        device = torch.device("cuda")
        print(f"INFO: Using GPU {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("WARN: Inferring on CPU")

    model = model_from_checkpoint(config, device)
    model = torch.compile(model)
    # onestep_errors = compute_onestep_errors(config, model, device)
    # print(
    #     f"One-step MSE: {np.mean(onestep_errors[:, 0])} ± {np.std(onestep_errors[:, 0])}"
    # )
    # print(
    #     f"One-step MAE: {np.mean(onestep_errors[:, 1])} ± {np.std(onestep_errors[:, 1])}"
    # )

    compute_rollouts(config, model, device)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--logging_folder", type=str, help="Use to override the config")
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
