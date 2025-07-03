# File:         main.py
# Date:         2024/11/12
# Description:  Top-level script to train Topological Deep Learning models

import argparse
import json
from os import listdir, makedirs, path as opath, scandir
from pathlib import Path
from sys import path as spath
from timeit import default_timer as timer
from typing import Callable

# Add the parent directory to the Python path
spath.append(opath.join(Path(__file__).parent.resolve(), ".."))

import numpy as np
from toponetx.classes import CombinatorialComplex
import torch
from torch.nn import MSELoss, L1Loss
from torch.optim import Adam
from torch.optim.lr_scheduler import ConstantLR, ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
import trimesh
import wandb

from data.movi_dataset import (
    MoviDataset,
    MoviNormalization,
    movi_dataset_collate_fn,
)
from models.hopnet import HOPNet
from models.ablations import HOPNet_NoSequential, HOPNet_NoObjectCells

from utils.plots import plot_errors, plot_errors_distribution
from utils.rollout import (
    build_base_complex,
    build_featured_complex,
    model_input_from_ccc,
    positions_from_model_output,
    shape_matching,
)


SCRIPT_DIR: str = opath.dirname(opath.realpath(__file__))

# PyTorch Configuration
torch.multiprocessing.set_sharing_strategy("file_system")
torch.set_float32_matmul_precision("high")

# Weights & Biases Configuration
RUN_ID_FILE: str = "wandb_run.json"  # Do not modify
WANDB_PROJECT: str = ""  # Fill it with your own project name
WANDB_ENTITY: str = ""  # Fill it with your own username

# Rollout Configuration
METADATA_FILENAME: str = "metadata.json"
COLLISIONS_FILENAME: str = "collisions.json"
MESHES_LOCATION: str = opath.join(SCRIPT_DIR, "..", "data", "objects")
MESH_FILENAME: str = "collision_geometry.obj"
ROLLOUT_START_TIMES: list[int] = [0]
ROLLOUT_DURATION: int = 102

# Horizon used in the dataset generation
H: int = 2


def get_model(
    model_name: str,
    num_channels: int,
    num_layers: int,
    activation_func: str,
    mlp_layers: int,
) -> torch.nn.Module:
    """Create a HOPNet model (or one if its ablation variant)

    Parameters
    ----------
    model_name : str
        Name of the model (one of [HOPNet, NoObjectCells, NoSequential])
    num_channels : int
        Hidden embedding size
    num_layers : int
        Number of message-passing layers
    activation_func : str
        Activation function to use inside MLPs
    mlp_layers : int
        Number of linear layers inside MLPs

    Returns
    -------
    torch.nn.Module
        HOPNet model (or one if its ablation variant)
    """
    # Basic validation
    assert num_channels > 0 and (num_channels % 2 == 0)
    assert num_layers > 0

    if activation_func.lower() == "relu":
        act_fn = torch.nn.ReLU
    elif activation_func.lower() == "selu":
        act_fn = torch.nn.SELU
    elif activation_func.lower() == "gelu":
        act_fn = torch.nn.GELU
    elif activation_func.lower() == "elu":
        act_fn = torch.nn.ELU
    else:
        raise RuntimeError(f"Unsupported activation function {activation_func}")

    if model_name == "HOPNet":
        model = HOPNet(
            in_channels=[
                H * (3 + 1) + 2 * (3 + 1),  # Nodes (velocity + center-mass)
                2 * 4,  # Edges (OG distance + norm, current distance + norm)
                3,  # Faces (normal vector)
                # Collision Faces (col vector, norm, 3x source face vectors, 3x target face vectors)
                4 + 2 * 3 * 4,
                H * (3 + 1) + 5,  # Objects (vel+norm, type, mass, friction, resitution)
            ],
            hid_channels=5 * [num_channels],
            num_layers=num_layers,
            activation_func=act_fn,
            mlp_layers=mlp_layers,
            out_channels=[3, 3],
        )
    elif model_name == "NoSequential":
        model = HOPNet_NoSequential(
            in_channels=[
                H * (3 + 1) + 2 * (3 + 1),  # Nodes (velocity + center-mass)
                2 * 4,  # Edges (OG distance + norm, current distance + norm)
                3,  # Faces (normal vector)
                # Collision Faces (col vector, norm, 3x source face vectors, 3x target face vectors)
                4 + 2 * 3 * 4,
                H * (3 + 1) + 5,  # Objects (vel+norm, type, mass, friction, resitution)
            ],
            hid_channels=5 * [num_channels],
            num_layers=num_layers,
            activation_func=act_fn,
            mlp_layers=mlp_layers,
            out_channels=[3, 3],
        )
    elif model_name == "NoObjectCells":
        model = HOPNet_NoObjectCells(
            in_channels=[
                H * (3 + 1) + 2 * (3 + 1) + (2 + 3 * 1),  # Nodes + object features
                2 * 4,  # Edges (OG distance + norm, current distance + norm)
                3,  # Faces (normal vector)
                # Collision Faces (col vector, norm, 3x source face vectors, 3x target face vectors)
                4 + 2 * 3 * 4,
            ],
            hid_channels=4 * [num_channels],
            num_layers=num_layers,
            activation_func=act_fn,
            mlp_layers=mlp_layers,
            out_channels=3,
        )
    else:
        raise RuntimeError(f"ERROR: Unknown model {model_name}")

    return model


def parameters_count(model: torch.nn.Module) -> int:
    """Get the number of learnable parameters inside a Pytorch model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: torch.nn.Module,
    device: torch.device,
    loader: DataLoader,
    norm: MoviNormalization,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable,
    log_file: str,
    writer: SummaryWriter,
    print_freq: int = 50,
    epoch_num: int = 0,
) -> np.ndarray:
    """Train a HOPNet model (or one of its ablation) for one epoch

    Parameters
    ----------
    moodel : torch.nn.Module
        Model to train
    device : torch.device
        Device to send the data to
    loader : DataLoader
        DataLoader containing the training set
    norm : MoviNormalization
        Normalization statistics
    optimizer : torch.optim.Optimizer
        Optimizer for backpropagation
    loss_fn : Callable
        Loss function to train the model with (from PyTorch)
    log_file : str
        Log file name (to dump performance metrics)
    writer: SummaryWriter
        Tensorboard SummaryWriter class for interactive logging
    print_freq : int, optional
        Number of batches between each log print, by default 50
    epoch_num : int, optional
        Used to print the epoch number in the terminal, by default 0

    Returns
    -------
    np.ndarray
        Array (B x 1) containing the training MSE loss for each training sample
    """
    # Put the NN model in training mode (to keep track of the gradient)
    model.train()

    losses = np.empty((len(loader), 1))

    # Profiling
    sum_data_loading_timings = 0
    sum_inference_timings = 0

    # Compute normalized values equivalent to zero acceleration (for static objects)
    t0_zero_acc = -1.0 * norm.t0_mean.clone().detach() / norm.t0_std
    t0_zero_acc = t0_zero_acc.to(device)
    t4_zero_acc = -1.0 * norm.t4_mean.clone().detach() / norm.t4_std
    t4_zero_acc = t4_zero_acc.to(device)

    train_iter = iter(loader)
    for i in tqdm(range(len(loader))):
        start = timer()

        # Manually iterate to profile the data loading
        input: dict = next(train_iter)
        # Get loss training masks
        x0_mask = (
            input["x0_mask"].unsqueeze(-1).to(device) if "x0_mask" in input else None
        )
        x4_mask = (
            input["x4_mask"].unsqueeze(-1).to(device) if "x4_mask" in input else None
        )
        # Move all big tensors at once on the GPU
        x0 = input["x0"].to(device)
        x1 = input["x1"].to(device)
        t0 = input["t0"].to(device)
        if not isinstance(model, HOPNet_NoObjectCells):
            x4 = input["x4"].to(device)
            t4 = input["t4"].to(device)
            b04 = input["b04"].to(device)

        if isinstance(model, HOPNet_NoSequential):
            a010 = input["a010"].to(device)
            a101 = input["a101"].to(device)
            b01 = input["b01"].to(device)
            b14 = input["b14"].to(device)
            m20 = torch.zeros((x0.shape[1], model.hid_channels[0]), device=x0.device)
            m21 = torch.zeros((x1.shape[1], model.hid_channels[1]), device=x0.device)
            m24 = torch.zeros((x4.shape[1], model.hid_channels[4]), device=x0.device)
        elif isinstance(model, HOPNet_NoObjectCells):
            a010 = input["a010"].to(device)

        sum_data_loading_timings += timer() - start

        # Reinitialize the loss for every sample
        loss = torch.tensor(0.0, device=device)

        # For each CC in the sample
        for time in range(x0.shape[0]):
            # Prepare the data for the model
            x2 = input["x2_l"][time].to(device) if input["x2_l"][time] != None else None
            x3 = input["x3_l"][time].to(device) if input["x3_l"][time] != None else None
            b02 = (
                input["b02_l"][time].to(device)
                if input["b02_l"][time] != None
                else None
            )
            b12 = (
                input["b12_l"][time].to(device)
                if input["b12_l"][time] != None
                else None
            )
            b23 = (
                input["b23_l"][time].to(device)
                if input["b23_l"][time] != None
                else None
            )
            if not isinstance(model, HOPNet_NoObjectCells):
                b24 = (
                    input["b24_l"][time].to(device)
                    if input["b24_l"][time] != None
                    else None
                )
            if isinstance(model, HOPNet_NoSequential):
                a232 = (
                    input["a232_l"][time].to(device)
                    if input["a232_l"][time] != None
                    else None
                )
                b03 = (
                    input["b03_l"][time].to(device)
                    if input["b03_l"][time] != None
                    else None
                )
                b13 = (
                    input["b13_l"][time].to(device)
                    if input["b13_l"][time] != None
                    else None
                )

            start = timer()

            # Infer the model
            if isinstance(model, HOPNet_NoSequential):
                out_0, out_4 = model(
                    x0[time],
                    x1[time],
                    x2,
                    x3,
                    x4[time],
                    t0_zero_acc,
                    t4_zero_acc,
                    a010,
                    a101,
                    a232,
                    b01,
                    b02,
                    b03,
                    b04,
                    b12,
                    b13,
                    b14,
                    b23,
                    b24,
                    m20,
                    m21,
                    m24,
                )
            elif isinstance(model, HOPNet_NoObjectCells):
                out_0 = model(
                    x0[time], x1[time], x2, x3, t0_zero_acc, a010, b02, b12, b23
                )
            else:
                out_0, out_4 = model(
                    x0[time],
                    x1[time],
                    x2,
                    x3,
                    x4[time],
                    t0_zero_acc,
                    t4_zero_acc,
                    b02,
                    b04,
                    b12,
                    b23,
                    b24,
                )

            sum_inference_timings += timer() - start

            # Compute the loss (automatically sum-aggregated)
            if x0_mask is not None:
                loss_fn = MSELoss(reduction="none")
                loss += (
                    torch.sum(loss_fn(out_0, t0[time]) * x0_mask[time]) / x0.shape[1]
                )
                if not isinstance(model, HOPNet_NoObjectCells):
                    loss += (
                        torch.sum(loss_fn(out_4, t4[time]) * x4_mask[time])
                        / x4.shape[1]
                    )
            else:
                loss += loss_fn(out_0, t0[time]) / x0.shape[1]
                if not isinstance(model, HOPNet_NoObjectCells):
                    loss += loss_fn(out_4, t4[time]) / x4.shape[1]

        # Track the loss for the complete sample
        losses[i, 0] = loss.item() / x0.shape[0]

        # Compute loss and gradients for the whole sample
        loss.backward()

        # Update the model weights
        optimizer.step()
        optimizer.zero_grad()

        # Tensorboard
        writer.add_scalar("train/loss", losses[i, 0], epoch_num * len(loader) + i)

        if i > 0 and i % print_freq == 0:
            print(
                f"{epoch_num:2}|{i}/{len(loader)} "
                f"D|C={sum_data_loading_timings/(i+1):.3f}|{sum_inference_timings/(i+1):.3f} [s] ; "
                f"Loss={losses[i, 0]:.6E}"
            )

        wandb.log(
            {
                "train/loss": losses[i, 0],
                "batch": epoch_num * len(loader) + i,
                "epoch": epoch_num,
            }
        )

    # Append to log file
    with open(log_file, "a+") as f:
        f.write(f"{epoch_num}: Loss={np.mean(losses):.6E} (std={np.std(losses):.6E})\n")

    print(f"{epoch_num}T: Loss={np.mean(losses):.6E} (std={np.std(losses):.6E})")

    return losses


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    device: torch.device,
    val_loader: DataLoader,
    norm: MoviNormalization,
    log_file: str,
    epoch_num: int = 0,
    export_outputs: bool = False,
    export_dir: str = "",
) -> np.ndarray:
    """Compute a model's performance on a validation set

    Parameters
    ----------
    model : torch.nn.Module
        Neural Network model to test
    device : torch.device
        Device to send the data to
    val_loader : DataLoader
        DataLoader containing the validation set
    norm : MoviNormalization
        Normalization statistics
    log_file : str
        Log file name (to dump performance metrics)
    export_outputs : bool
        True to export the model outputs to dictionaries, False to discard them
    export_dir : str
        Path to the directory to export model outputs

    Returns
    -------
    np.ndarray
        Array (N x 4) containing the MAE (0), MSE (1), MAE standardized (2), and
        MSE standardized (3) metrics for each sample
    """
    # Put the NN model in inference mode (not keeping track of the gradient)
    model.eval()

    mae_fn = L1Loss(reduction="sum")
    mse_fn = MSELoss(reduction="sum")
    metrics = np.empty((len(val_loader), 4))

    # Profiling
    sum_data_loading_timings = 0
    sum_inference_timings = 0

    # Compute normalized values equivalent to zero acceleration (for static objects)
    t0_zero_acc = -1.0 * norm.t0_mean.clone().detach() / norm.t0_std
    t0_zero_acc = t0_zero_acc.to(device)
    t4_zero_acc = -1.0 * norm.t4_mean.clone().detach() / norm.t4_std
    t4_zero_acc = t4_zero_acc.to(device)

    val_iter = iter(val_loader)
    for i in tqdm(range(len(val_loader))):
        start = timer()

        # Manually iterate to profile the data loading
        input: dict = next(val_iter)
        # Move all big tensors at once on the GPU
        x0 = input["x0"].to(device)
        x1 = input["x1"].to(device)
        t0 = input["t0"].to(device)
        if not isinstance(model, HOPNet_NoObjectCells):
            x4 = input["x4"].to(device)
            t4 = input["t4"].to(device)
            b04 = input["b04"].to(device)

        if isinstance(model, HOPNet_NoSequential):
            a010 = input["a010"].to(device)
            a101 = input["a101"].to(device)
            b01 = input["b01"].to(device)
            b14 = input["b14"].to(device)
            m20 = torch.zeros((x0.shape[1], model.hid_channels[0]), device=x0.device)
            m21 = torch.zeros((x1.shape[1], model.hid_channels[1]), device=x0.device)
            m24 = torch.zeros((x4.shape[1], model.hid_channels[4]), device=x0.device)
        elif isinstance(model, HOPNet_NoObjectCells):
            a010 = input["a010"].to(device)

        sum_data_loading_timings += timer() - start

        # Reinitialize the loss for every timestep
        mae_loss = torch.tensor(0.0, device=device)
        mse_loss = torch.tensor(0.0, device=device)
        mae_loss_standardized = torch.tensor(0.0, device=device)
        mse_loss_standardized = torch.tensor(0.0, device=device)

        out0_l: list = []
        out4_l: list = []

        # For each CC in the sample
        for time in range(x0.shape[0]):
            # Prepare the data for the model
            x2 = input["x2_l"][time].to(device) if input["x2_l"][time] != None else None
            x3 = input["x3_l"][time].to(device) if input["x3_l"][time] != None else None
            b02 = (
                input["b02_l"][time].to(device)
                if input["b02_l"][time] != None
                else None
            )
            b12 = (
                input["b12_l"][time].to(device)
                if input["b12_l"][time] != None
                else None
            )
            b23 = (
                input["b23_l"][time].to(device)
                if input["b23_l"][time] != None
                else None
            )
            if not isinstance(model, HOPNet_NoObjectCells):
                b24 = (
                    input["b24_l"][time].to(device)
                    if input["b24_l"][time] != None
                    else None
                )
            if isinstance(model, HOPNet_NoSequential):
                a232 = (
                    input["a232_l"][time].to(device)
                    if input["a232_l"][time] != None
                    else None
                )
                b03 = (
                    input["b03_l"][time].to(device)
                    if input["b03_l"][time] != None
                    else None
                )
                b13 = (
                    input["b13_l"][time].to(device)
                    if input["b13_l"][time] != None
                    else None
                )

            start = timer()

            # Infer the model
            if isinstance(model, HOPNet_NoSequential):
                out_0, out_4 = model(
                    x0[time],
                    x1[time],
                    x2,
                    x3,
                    x4[time],
                    t0_zero_acc,
                    t4_zero_acc,
                    a010,
                    a101,
                    a232,
                    b01,
                    b02,
                    b03,
                    b04,
                    b12,
                    b13,
                    b14,
                    b23,
                    b24,
                    m20,
                    m21,
                    m24,
                )
            elif isinstance(model, HOPNet_NoObjectCells):
                out_0 = model(
                    x0[time], x1[time], x2, x3, t0_zero_acc, a010, b02, b12, b23
                )
            else:
                out_0, out_4 = model(
                    x0[time],
                    x1[time],
                    x2,
                    x3,
                    x4[time],
                    t0_zero_acc,
                    t4_zero_acc,
                    b02,
                    b04,
                    b12,
                    b23,
                    b24,
                )

            sum_inference_timings += timer() - start

            # Compute the loss (automatically sum-aggregated)
            mse_loss += mse_fn(out_0, t0[time]) / x0.shape[1]
            if not isinstance(model, HOPNet_NoObjectCells):
                mse_loss += mse_fn(out_4, t4[time]) / x4.shape[1]

            mae_loss += mae_fn(out_0, t0[time]) / x0.shape[1]
            if not isinstance(model, HOPNet_NoObjectCells):
                mae_loss += mae_fn(out_4, t4[time]) / x4.shape[1]

            # Compute the normalized loss (automatically sum-aggregated)
            mae_loss_standardized += (
                mae_fn(out_0, t0[time]) / x0.shape[1] * norm.t0_std[0]
            )
            if not isinstance(model, HOPNet_NoObjectCells):
                mae_loss_standardized += (
                    mae_fn(out_4, t4[time]) / x4.shape[1] * norm.t4_std[0]
                )
            mse_loss_standardized += (
                mse_fn(out_0, t0[time]) / x0.shape[1] * norm.t0_std[0]
            )
            if not isinstance(model, HOPNet_NoObjectCells):
                mse_loss_standardized += (
                    mse_fn(out_4, t4[time]) / x4.shape[1] * norm.t4_std[0]
                )

            if export_outputs:
                out0_l.append(out_0.detach().cpu().numpy())
                out4_l.append(out_4.detach().cpu().numpy())

        # Normalize by number of timesteps
        metrics[i, 0] = mae_loss.item() / x0.shape[0]
        metrics[i, 1] = mse_loss.item() / x0.shape[0]
        metrics[i, 2] = mae_loss_standardized.item() / x0.shape[0]
        metrics[i, 3] = mse_loss_standardized.item() / x0.shape[0]

        # Export the output
        if export_outputs:
            np.save(
                opath.join(export_dir, f"{str(input['seed'])}.npy"),
                {"out0_l": out0_l, "out4_l": out4_l},  # type: ignore
            )

    # Append to log file
    with open(log_file, "a+") as f:
        f.write(
            f"{epoch_num}: MAE={np.mean(metrics[:,0]):.6E} (std={np.std(metrics[:,0]):.6E}) "
            f"MSE={np.mean(metrics[:,1]):.6E} (std={np.std(metrics[:,1]):.6E})\n"
            f"{epoch_num}: MAE standardized={np.mean(metrics[:,2]):.6E} (std={np.std(metrics[:,2]):.6E}) "
            f"MSE standardized={np.mean(metrics[:,3]):.6E} (std={np.std(metrics[:,3]):.6E})\n"
        )

    print(
        f"{epoch_num}V: MAE={np.mean(metrics[:,0]):.6E} (std={np.std(metrics[:,0]):.6E}) "
        f"MSE={np.mean(metrics[:,1]):.6E} (std={np.std(metrics[:,1]):.6E})"
        f"{epoch_num}V: MAE stand.={np.mean(metrics[:,2]):.6E} (std={np.std(metrics[:,2]):.6E}) "
        f"MSE stand.={np.mean(metrics[:,3]):.6E} (std={np.std(metrics[:,3]):.6E})\n"
    )

    return metrics


def get_meshes(objects: list[dict]) -> list[trimesh.Trimesh]:
    """Get the meshes corresponding to a list of objects (no transformation applied)

    Parameters
    ----------
    objects : list[dict]
        Metadata of required objects

    Returns
    -------
    list[trimesh.Trimesh]
        Corresponding meshes (same order as objects)
    """

    def get_mesh(shape: str) -> trimesh.Trimesh:
        mesh_path: str = opath.join(MESHES_LOCATION, shape, MESH_FILENAME)

        if opath.exists(mesh_path):
            return trimesh.load(mesh_path)
        else:
            raise RuntimeError(f"Unknown shape {shape}")

    meshes: list[trimesh.Trimesh] = []
    for object in objects:
        scale: float = object["size"]
        mesh: trimesh.Trimesh = get_mesh(object["shape"])
        mesh.apply_scale(scale)
        meshes.append(mesh)

    return meshes


@torch.no_grad
def individual_rollout(
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
    norm: MoviNormalization,
    objects: list[dict],
    meshes: list[trimesh.Trimesh],
    base_ccc: CombinatorialComplex,
    gt_collisions: dict | None,
    triangle_ids: list[np.ndarray],
    obj_idx_from_node_idx: dict,
    start_time: int,
    duration: int,
    collision_radius: float,
) -> np.ndarray:
    """Perform rollout of a certain duration starting from a unique timestep

    Returns
    -------
    np.ndarray
        Rollout trajectory ; [duration, objects_count, 3 + 4]
        3 position (x, y, z) + 4 orientation (w, x, y, z)
    """

    # Initial object positions and orientations
    obj_pos = np.array(
        [obj["positions"][start_time : start_time + H + 2] for obj in objects]
    )
    obj_pos = np.moveaxis(obj_pos, 0, 1)  # Shape [timesteps, obj_count, 3]
    obj_quat = np.array(
        [obj["quaternions"][start_time : start_time + H + 2] for obj in objects]
    )
    obj_quat = np.moveaxis(obj_quat, 0, 1)  # Shape [timesteps, obj_count, 4]
    invalid_collisions_ratios = np.zeros((duration + H + 2, 1))

    # Use pre-computed collisions for the first timestep if available
    collisions = (
        gt_collisions[str(start_time + H + 1)] if gt_collisions is not None else None
    )

    # Rollout
    for t in range(H + 1, H + 1 + duration):
        # Step 1: Create the CCC
        ccc, nodes_pos, invalid_collisions_ratio = build_featured_complex(
            base_ccc,
            collisions,
            triangle_ids,
            obj_idx_from_node_idx,
            obj_pos[t - H : t + 1],
            obj_quat[t - H : t + 1],
            objects,
            meshes,
            model_name == "NoObjectCells",
            H,
            collision_radius,
        )
        invalid_collisions_ratios[t] = invalid_collisions_ratio

        # Step 2: Model inference
        with torch.no_grad():
            model.eval()
            out = model(
                *model_input_from_ccc(
                    ccc,
                    horizon=H,
                    norm=norm,
                    model=model,
                    model_name=model_name,
                    device=device,
                )
            )
            if model_name == "NoObjectCells":
                # Strip virtual object nodes
                out0 = out[: -len(objects), :]
            else:
                out0 = out[0]

        # Step 3: Compute predicted positions from accelerations
        pred_nodes_pos = positions_from_model_output(out0, norm, nodes_pos[-2:])

        # Step 4: Compute shape matching for the new object rotation
        new_obj_pos, new_obj_quat = shape_matching(pred_nodes_pos, meshes)

        # Step 5: Update the obj_pos and obj_quat with the predicted positions
        obj_pos = np.concatenate((obj_pos, np.expand_dims(new_obj_pos, 0)), axis=0)
        obj_quat = np.concatenate((obj_quat, np.expand_dims(new_obj_quat, 0)), axis=0)

        # Only use pre-computed collisions for the first timestep
        collisions = None

    invalid_collisions_ratios = np.expand_dims(
        np.repeat(invalid_collisions_ratios, obj_pos.shape[1], axis=-1), -1
    )

    return np.concatenate((obj_pos, obj_quat, invalid_collisions_ratios), axis=-1)


@torch.no_grad
def rollout_sample(
    model: torch.nn.Module,
    model_name: str,
    device: torch.device,
    sample: str,
    norm: MoviNormalization,
    collision_radius: float,
) -> np.ndarray:
    """Perform rollout on a single dataset sample

    Returns
    -------
    np.ndarray
        Rollout trajectories [<ROLLOUT_START_TIMES>, duration, objects_count, 3 + 4]
    """
    # Load the sample metadata
    metadata_path: str = opath.join(sample, METADATA_FILENAME)
    with open(metadata_path) as f:
        metadata: dict = json.load(f)
    objects: list[dict] = metadata["instances"]
    num_frames: int = metadata["metadata"]["num_frames"]

    # Convert the sample as meshes
    meshes = get_meshes(objects)

    # Add the floor to the scene (not generated by Kubric by default)
    FLOOR_FRICTION: float = 0.30  # Constant from Kubric
    FLOOR_RESTITUTION: float = 0.50  # Constant from Kubric
    FLOOR_SIZE: float = 20.0  # Minimal approximation, Kubric uses 40

    FLOOR = trimesh.Trimesh(
        vertices=[
            [-FLOOR_SIZE, -FLOOR_SIZE, 0],
            [-FLOOR_SIZE, FLOOR_SIZE, 0],
            [FLOOR_SIZE, -FLOOR_SIZE, 0],
            [FLOOR_SIZE, FLOOR_SIZE, 0],
        ],
        faces=[[0, 1, 2], [1, 2, 3]],
    )
    FLOOR_META = {
        "asset_id": "floor",
        "angular_velocities": np.zeros((num_frames, 3)).tolist(),
        "friction": FLOOR_FRICTION,
        "mass": 0.0,
        "positions": np.zeros((num_frames, 3)).tolist(),
        "quaternions": np.repeat([[1.0, 0.0, 0.0, 0.0]], num_frames, axis=0).tolist(),
        "restitution": FLOOR_RESTITUTION,
        "size": 1.0,
        "velocities": np.zeros((num_frames, 3)).tolist(),
    }
    objects.append(FLOOR_META)
    meshes.append(FLOOR)

    # If available, load the precomputed ground truth collisions
    if opath.exists(opath.join(sample, COLLISIONS_FILENAME)):
        with open(opath.join(sample, COLLISIONS_FILENAME), "r") as f:
            gt_collisions = json.load(f)
    else:
        gt_collisions = None

    # Compute all preliminary requirements for rollout
    base_ccc, triangle_ids, obj_idx_from_node_idx = build_base_complex(
        meshes, objects, isinstance(model, HOPNet_NoObjectCells)
    )
    for obj in objects:
        obj["positions"] = np.array(obj["positions"]).tolist()
        obj["quaternions"] = np.array(obj["quaternions"]).tolist()

    # Store the model output when starting rollout at different times
    # 3 = position (x, y, z) ; 4 = quaternion (w, x, y, z) ; 1 (penetration ratio)
    model_output = np.empty(
        (len(ROLLOUT_START_TIMES), ROLLOUT_DURATION + H + 2, len(objects), 3 + 4 + 1)
    )

    for idx, start_time in enumerate(ROLLOUT_START_TIMES):
        model_output[idx, :, :, :] = individual_rollout(
            model,
            model_name,
            device,
            norm,
            objects,
            meshes,
            base_ccc,
            gt_collisions,
            triangle_ids,
            obj_idx_from_node_idx,
            start_time,
            ROLLOUT_DURATION,
            collision_radius,
        )

    # Remove the floor predictions (as it's always static and not useful for errors)
    return model_output[:, :, :-1, :]


def get_dataloaders_training(
    dataset_dir: str,
    normalization_file: str,
    nox4: bool,
    dataloader_options: dict,
) -> tuple[DataLoader, DataLoader, MoviNormalization]:
    """Load a training and validation set from a dataset

    A fixed training/validation split will automatically be loaded.

    Parameters
    ----------
    dataset_dir : str
        Location of the dataset
    normalization_file : str
        Absolute path of the normalization file
    nox4 : bool
        True to load combinatorial complexes for ablation model NoObjectCells
    dataloader_options : dict
        Options for the Pytorch DataLoader

    Returns
    -------
    tuple[DataLoader, DataLoader, MoviNormalization]
        Training and Validation DataLoader, Normalization statistics
    """
    # Load the dataset
    train_set = MoviDataset(dataset_dir, "train", normalization_file, nox4)
    val_set = MoviDataset(dataset_dir, "val", normalization_file, nox4)

    # Create DataLoader from the Dataset objects
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        collate_fn=movi_dataset_collate_fn,
        **dataloader_options,
    )
    val_loader = DataLoader(
        val_set,
        collate_fn=movi_dataset_collate_fn,
        **dataloader_options,
    )

    return train_loader, val_loader, train_set.normalization


def train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)

    # Forced CPU training or automatic GPU selection
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        cuda_device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(cuda_device_idx)
        print(f"CUDA {cuda_device_idx}: {device_name}")
    elif device.type == "mps":
        device_name = "MPS"
        print(f"MPS backend")
    else:
        device_name = "CPU"
        print(f"WARNING: Training on CPU")

    # Check the learning rate parameters
    if args.lr <= 0:
        print(f"ERROR: learning rate must be > 0")
        return 1
    learning_rate = args.lr

    # Set the Pytorch DataLoader options
    dataloader_options = {
        "batch_size": 1,  # Gradient accumulation is performed manually
        "persistent_workers": False,  # Avoid memory leaks
        "num_workers": args.workers,
        "pin_memory": False,  # Cannot pin SparseCUDA tensors
    }

    # Load the datasets as DataLoaders
    train_loader, val_loader, norm = get_dataloaders_training(
        args.dataset_dir,
        args.normalization_file,
        args.model == "NoObjectCells",
        dataloader_options,
    )

    # Declare the NN model
    model = get_model(
        args.model, args.channels, args.layers, args.activation_func, args.mlp_layers
    )
    model = model.to(device)
    print("INFO: Model parameters:", parameters_count(model))

    # Declare optimizer and loss
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = MSELoss(reduction="sum")

    # Declare learning rate scheduler
    if args.lr_exp_decay > 0:
        lr_scheduler = ExponentialLR(optimizer, gamma=args.lr_exp_decay)
    else:
        lr_scheduler = ConstantLR(optimizer, factor=1.0, total_iters=1)

    # If the log directory is not empty, another run can be resumed
    wandb_run_id = None
    if len(listdir(args.log_dir)) > 0:
        print(f"DEBUG: Log directory {args.log_dir} not empty")
        if opath.isfile(opath.join(args.log_dir, RUN_ID_FILE)):
            with open(opath.join(args.log_dir, RUN_ID_FILE), "r") as f:
                wandb_run_id = json.load(f)["run_id"]
            print(f"INFO: Found run ID '{wandb_run_id}' in {args.log_dir}")

        # List the current checkpoints in the run directory (if any)
        checkpoint_prefix = (
            f"{args.model}_c{args.channels}_l{args.layers}_mlp{args.mlp_layers}_e"
        )
        checkpoints = [
            f for f in listdir(args.log_dir) if f.startswith(checkpoint_prefix)
        ]
        checkpoints = sorted(checkpoints, key=lambda f: int(f[-5:-3]))
        # Grab the latest checkpoint if possible
        if len(checkpoints) > 0:
            args.checkpoint = checkpoints[-1]
            print(f"INFO: Found checkpoint '{args.checkpoint}' in {args.log_dir}")
        else:
            print(f"WARN: No checkpoint found in {args.log_dir} ; training from zero")
            args.checkpoint = None
    else:
        print(f"DEBUG: Log directory is empty, no run to resume")

    # If a checkpoint was found, resume the model training from it
    if args.checkpoint != None and args.checkpoint != "":
        print(f"INFO: Loading checkpoint '{args.checkpoint}' ...")
        weights = torch.load(
            opath.join(args.log_dir, args.checkpoint), map_location=device
        )
        model.load_state_dict(weights, strict=True)
        # Determine previous epoch from model name
        starting_epoch: int = int(args.checkpoint[-5:-3]) + 1
        # Advance the learning rate scheduler
        optimizer.step()  # To avoid Pytorch warning
        [lr_scheduler.step() for _ in range(starting_epoch)]
    else:
        starting_epoch: int = 0

    # Declare log files
    log_file_val = opath.join(args.log_dir, "val.txt")
    log_file_train = opath.join(args.log_dir, "train.txt")

    wandb_config = {
        "batch_size": 1,
        "model": args.model,
        "dataset": opath.basename(opath.normpath(args.dataset_dir)),
        "channels": args.channels,
        "layers": args.layers,
        "mlp_layers": args.mlp_layers,
        "activation_func": args.activation_func.lower(),
        "epochs": args.epochs,
        "loss": "MSE",
        "optimizer": "adam",
        "lr": learning_rate,
        "train_batches": len(train_loader),
        "val_items": len(val_loader),
        "workers": args.workers,
        "gpu": device_name,
    }

    if wandb_run_id is not None:
        print(f"DEBUG: Resuming W&B run '{wandb_run_id}'")
        # Resume the existing run
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            id=wandb_run_id,
            resume="allow",
            mode="online" if args.online else "disabled",
        )
    else:
        # Create a new W&B run
        run = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            job_type="train",
            config=wandb_config,
            mode="online" if args.online else "disabled",
        )
        wandb_run_id = run.id
        with open(opath.join(args.log_dir, RUN_ID_FILE), "w") as f:
            json.dump({"run_id": wandb_run_id}, f)
        print(f"DEBUG: Created W&B run '{wandb_run_id}'")

    writer_name = f"{SCRIPT_DIR}/runs/{wandb_run_id}"

    # Configure X-axes for train/ and val/ metrics
    wandb.define_metric("batch")  # Create a custom X-ax
    wandb.define_metric("epoch")  # Create a custom X-axis
    wandb.define_metric("train/*", summary="last", step_metric="batch")
    wandb.define_metric("train/lr", summary="last", step_metric="epoch")
    wandb.define_metric("val/*", summary="last", step_metric="epoch")
    # Watch the gradients in the Pytorch model
    wandb.watch(model) if args.online else None

    # Tensorboard configuration
    writer = SummaryWriter(writer_name)

    # Train the NN model
    for epoch in range(starting_epoch, args.epochs):
        writer.add_scalar("train/lr", lr_scheduler.get_last_lr()[0], epoch)
        losses = train_one_epoch(
            model,
            device,
            train_loader,
            norm,
            optimizer,
            loss_fn,
            log_file=log_file_train,
            writer=writer,
            epoch_num=epoch,
        )

        # Compute performance on the validation set
        metrics = validate(
            model,
            device,
            val_loader,
            norm,
            log_file=log_file_val,
            epoch_num=epoch,
        )
        wandb.log(
            {
                "train/lr": lr_scheduler.get_last_lr()[0],
                "val/mae": np.mean(metrics[:, 0]),
                "val/mae_std": np.std(metrics[:, 0]),
                "val/mse": np.mean(metrics[:, 1]),
                "val/mse_std": np.std(metrics[:, 1]),
                "val/mae_standardized": np.mean(metrics[:, 2]),
                "val/mae_standardized_std": np.std(metrics[:, 2]),
                "val/mse_standardized": np.mean(metrics[:, 3]),
                "val/mse_standardized_std": np.std(metrics[:, 3]),
                "epoch": epoch,
            }
        )
        writer.add_scalar("val/mae", np.mean(metrics[:, 0]), epoch)
        writer.add_scalar("val/mse", np.mean(metrics[:, 1]), epoch)
        writer.add_scalar("val/mae_standardized", np.mean(metrics[:, 2]), epoch)
        writer.add_scalar("val/mse_standardized", np.mean(metrics[:, 3]), epoch)
        writer.add_scalar("val/mae_std", np.std(metrics[:, 0]), epoch)
        writer.add_scalar("val/mse_std", np.std(metrics[:, 1]), epoch)
        writer.add_scalar("val/mae_standardized_std", np.std(metrics[:, 2]), epoch)
        writer.add_scalar("val/mse_standardized_std", np.std(metrics[:, 3]), epoch)

        # Save a checkpoint for the model
        torch.save(
            model.state_dict(),
            opath.join(
                args.log_dir,
                f"{args.model}_c{args.channels}_l{args.layers}_mlp{args.mlp_layers}_e{epoch:02d}.pt",
            ),
        )

        lr_scheduler.step()

    # Tensorboard hparams
    writer.add_hparams(
        wandb_config,
        {
            "train/fin_loss": np.mean(losses[:, 0]),
            "train/fin_loss_std": np.std(losses[:, 0]),
            "val/fin_mae": np.mean(metrics[:, 0]),
            "val/fin_mae_std": np.std(metrics[:, 0]),
            "val/fin_mse": np.mean(metrics[:, 1]),
            "val/fin_mse_std": np.std(metrics[:, 1]),
            "val/fin_mae_standardized": np.mean(metrics[:, 2]),
            "val/fin_mae_standardized_std": np.std(metrics[:, 2]),
            "val/fin_mse_standardized": np.mean(metrics[:, 3]),
            "val/fin_mse_standardized_std": np.std(metrics[:, 3]),
        },
        run_name=".",
    )

    writer.close()

    return 0


@torch.no_grad()
def evaluate_rollout(args: argparse.Namespace) -> int:
    if not args.checkpoint:
        print("ERROR: Select a checkpoint as CLI argument with --checkpoint=...")
        return 1

    # Forced CPU inference or automatic GPU selection
    if args.cpu:
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        cuda_device_idx = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(cuda_device_idx)
        print(f"CUDA {cuda_device_idx}: {device_name}")
    elif device.type == "mps":
        device_name = "MPS"
        print(f"MPS backend")
    else:
        device_name = "CPU"
        print(f"WARNING: Inferring on CPU")

    # Load the dataset (to get the raw samples indices and normalization)
    dataset = MoviDataset(
        args.dataset_dir,
        "test",
        args.normalization_file,
        args.model == "NoSequential",
        include_not_preprocessed=True,
    )
    samples = sorted(dataset.samples_paths, key=lambda e: int(opath.basename(e)))
    norm: MoviNormalization = dataset.normalization

    # Declare the NN model
    model = get_model(
        args.model, args.channels, args.layers, args.activation_func, args.mlp_layers
    )
    model = model.to(device)
    print(model)

    # Load the pre-trained checkpoint
    weights = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(weights, strict=True)

    # Auto-regressive rollout the model on the testing dataset (if not already computed)
    for sample in tqdm(samples):
        sample_id = int(opath.basename(sample))
        rollout_output_file = opath.join(args.log_dir, f"{sample_id}.npy")
        if not opath.exists(rollout_output_file):
            output = rollout_sample(
                model, args.model, device, sample, norm, args.collision_radius
            )
            np.save(opath.join(args.log_dir, str(sample_id)), output)

    return 0


def compute_errors(args: argparse.Namespace) -> int:
    # Make sure the directories exist
    assert opath.exists(args.dataset_dir)
    assert opath.exists(args.log_dir)

    # List the NPY rollout trajectories in args.log_dir
    samples_npy = [
        f.name
        for f in scandir(args.log_dir)
        if (f.is_file() and f.name.lower().endswith(".npy") and f.name != "rollout.npy")
    ]
    samples_npy = sorted(samples_npy, key=lambda e: int(Path(e).stem))

    if len(samples_npy) == 0:
        print(f"ERROR: Directory {args.log_dir} does not contain any NPY files")
        return 1

    # Compute the rollout error for each sample
    all_errors = np.zeros(
        (len(samples_npy), len(ROLLOUT_START_TIMES), 5, ROLLOUT_DURATION + H + 2)
    )
    for exp_idx, exp in enumerate(tqdm(samples_npy)):
        # Load the computed rollout trajectory
        traj = np.load(opath.join(args.log_dir, exp))

        # Load the ground truth data
        with open(opath.join(args.dataset_dir, Path(exp).stem, METADATA_FILENAME)) as f:
            metadata: dict = json.load(f)
        objects: list[dict] = metadata["instances"]
        obj_pos = np.moveaxis(np.array([obj["positions"] for obj in objects]), 0, 1)
        obj_quat = np.moveaxis(np.array([obj["quaternions"] for obj in objects]), 0, 1)

        # Minimal input validation
        assert traj.shape[2] == len(objects)
        assert traj.shape[1] == ROLLOUT_DURATION + H + 2

        pos_rmse = np.empty((len(ROLLOUT_START_TIMES), traj.shape[1], 3))
        ori_rmse = np.empty((len(ROLLOUT_START_TIMES), traj.shape[1], 4))
        # For each starting time, compute the errors
        for start_time_idx, start_time in enumerate(ROLLOUT_START_TIMES):
            # Compute position RMSE and MAE
            pos_err = (
                traj[start_time_idx, :, :, :3]
                - obj_pos[start_time : start_time + traj.shape[1]]
            )
            pos_rmse = np.sqrt(np.mean(np.linalg.norm(pos_err, axis=-1) ** 2, axis=-1))
            pos_mae = np.mean(np.linalg.norm(pos_err, axis=-1), axis=-1)
            all_errors[exp_idx, start_time_idx, 0] = pos_mae
            all_errors[exp_idx, start_time_idx, 1] = pos_rmse

            # Compute orientation RMSE and MAE
            traj[start_time_idx, :, :, 4:7] *= -1  # Compute quaternion conjugate
            ori_err = 2 * np.arcsin(
                np.linalg.norm(
                    quat_multiply(
                        traj[start_time_idx, :, :, 3:7],
                        obj_quat[start_time : start_time + traj.shape[1]],
                    )[:, :, 1:],
                    axis=-1,
                )
            )
            ori_err *= 360 / (2 * np.pi)
            ori_rmse = np.sqrt(np.nanmean(ori_err**2, axis=-1))
            ori_mae = np.nanmean(ori_err, axis=-1)
            all_errors[exp_idx, start_time_idx, 2] = ori_mae
            all_errors[exp_idx, start_time_idx, 3] = ori_rmse
            # If invalid collisions count is available (only for HOPNet, not FIGNet)
            if traj.shape[-1] == 8:
                all_errors[exp_idx, start_time_idx, 4] = traj[start_time_idx, :, 0, 7]

    np.save(opath.join(args.log_dir, "rollout"), all_errors)
    mean_errors = np.mean(all_errors, axis=(0, 1))
    std_errors = np.std(all_errors, axis=(0, 1))
    print(f"Pos RMSE @ 25 steps\t: {mean_errors[1,25]:.6f} ± {std_errors[1,25]:.6f}")
    print(f"Pos RMSE @ 50 steps\t: {mean_errors[1,50]:.6f} ± {std_errors[1,50]:.6f}")
    print(f"Pos RMSE @ 75 steps\t: {mean_errors[1,75]:.6f} ± {std_errors[1,75]:.6f}")
    print(f"Pos RMSE @ 100 steps\t: {mean_errors[1,100]:.6f} ± {std_errors[1,100]:.6f}")
    print(f"Ori RMSE @ 25 steps\t: {mean_errors[2,25]:.6f} ± {std_errors[2,25]:.6f}")
    print(f"Ori RMSE @ 50 steps\t: {mean_errors[2,50]:.6f} ± {std_errors[2,50]:.6f}")
    print(f"Ori RMSE @ 75 steps\t: {mean_errors[2,75]:.6f} ± {std_errors[2,75]:.6f}")
    print(f"Ori RMSE @ 100 steps\t: {mean_errors[2,100]:.6f} ± {std_errors[2,100]:.6f}")
    plot_errors(all_errors).write_image(opath.join(args.log_dir, "rollout_errors.png"))
    plot_errors_distribution(all_errors).write_image(
        opath.join(args.log_dir, "rollout_errors_distribution.png")
    )

    return 0


def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = np.split(quaternion0, 4, axis=-1)
    w1, x1, y1, z1 = np.split(quaternion1, 4, axis=-1)
    return np.concatenate(
        (
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
        ),
        axis=-1,
    )


def main(args: argparse.Namespace) -> int:
    # Create the target log directory if it doesn't exist
    makedirs(args.log_dir, exist_ok=True)

    if args.rollout:
        ret = evaluate_rollout(args)
        if ret != 0:
            return ret
        return compute_errors(args)
    elif args.errors:
        return compute_errors(args)
    else:
        args.checkpoint = None  # Checkpoint selection will be automatic
        return train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="High-Order topological Physics-informed Network"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Location of the MoVi dataset",
    )
    parser.add_argument(
        "--model",
        default="HOPNet",
        type=str,
        help="One of [HOPNet, NoObjectCells, NoSequential]",
    )
    parser.add_argument(
        "--layers", default=1, type=int, help="Number of message-passing layers"
    )
    parser.add_argument(
        "--activation_func", default="relu", type=str, help="MLPs activation function"
    )
    parser.add_argument(
        "--mlp_layers", default=2, type=int, help="Number of linear layers in MLPs"
    )
    parser.add_argument(
        "--channels", default=128, type=int, help="Size of hidden embeddings"
    )
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate (LR)")
    parser.add_argument(
        "--lr_exp_decay",
        default=0,  # Use 0.9441 for 40 epochs exp decay by 10x
        type=float,
        help="Learning rate exponential decay gamma coefficient",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        help="Location of the log directory",
        required=True,
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        help="Number of DataLoader workers",
    )
    parser.add_argument(
        "--epochs",
        default=40,
        type=int,
        help="Numer of epochs (training only)",
    )
    parser.add_argument(
        "--normalization_file",
        type=str,
        default="",
        help="Path to the normalization file (.npy)",
    )
    parser.add_argument(
        "--checkpoint", type=str, help="Location of the pre-trained model"
    )
    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU training (no GPU)"
    )
    parser.add_argument(
        "--online", action="store_true", help="Log this run to Weights & Biases"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed (for training only)"
    )
    parser.add_argument(
        "--collision_radius",
        type=float,
        default=0.1,
        help="Collision radius d_c (for autoregressive rollout only)",
    )
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--rollout",
        default=False,
        action="store_true",
        help="Inference (no training) with auto-regressive rollout",
    )
    grp.add_argument(
        "--errors",
        default=False,
        action="store_true",
        help="Compute auto-regressive rollout errors",
    )
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
