# MIT License
#
# Copyright (c) [2024] [Zongyao Yi]
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import os

import numpy as np
import torch
import torch.utils
import torch.utils.data
import tqdm
import yaml
from torchvision import transforms as T
import wandb

from fignet.data_loader import MoviDataset, MujocoDataset, ToTensor, collate_fn
from fignet.logger import Logger
from fignet.plt_utils import init_fig, plot_grad_flow
from fignet.scene import Scene
from fignet.simulator import LearnedSimulator
from fignet.types import KinematicType, NodeType
from fignet.utils import (
    optimizer_to,
    orientation_error_deg,
    parameters_count,
    rollout,
    to_numpy,
    visualize_trajectory,
)

# W&B Configuration
RUN_FILE: str = "wandb_run.json"


class Trainer:
    """Trainer for the GNN-based simulator"""

    def __init__(
        self,
        sim: LearnedSimulator,
        config: dict,
        logger: Logger,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        torch.manual_seed(config["seed"])
        self._device = device
        self._sim = sim
        if config.get("data_config"):
            self._input_seq_length = config["data_config"].get("input_seq_length", 3)
        else:
            self._input_seq_length = 3
        data_path = config["data_path"]
        test_data_path = config["test_data_path"]
        batch_size = config.get("batch_size", 1)
        self._datasets = {}
        self._dataset_type = config["dataset_type"]
        if config["dataset_type"] == "Mujoco":
            dataset_builder = MujocoDataset
        elif config["dataset_type"] == "Movi":
            dataset_builder = MoviDataset
        else:
            raise RuntimeError(f"ERROR: Unknown dataset type {config['dataset_type']}")
        logger.print(f"Model parameters: {parameters_count(sim)}")
        self._datasets["train"] = dataset_builder(
            data_path,
            input_sequence_length=self._input_seq_length,
            split="train",
            mode="sample",
            transform=T.Compose([ToTensor()]),
            config=config.get("data_config"),
        )
        self._datasets["rollout"] = dataset_builder(
            test_data_path,
            input_sequence_length=config["rollout_steps"],
            split="val",
            mode="trajectory",
            transform=T.Compose([ToTensor(self._device)]),
            config=config.get("data_config"),
        )
        self._datasets["val"] = dataset_builder(
            test_data_path,
            input_sequence_length=self._input_seq_length,
            split="val",
            mode="sample",
            transform=T.Compose([ToTensor()]),
            config=config.get("data_config"),
        )
        self._dataloaders = {}
        self._dataloaders["train"] = torch.utils.data.DataLoader(
            self._datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=config.get("num_workers", 1),
            pin_memory=True,
            collate_fn=collate_fn,
        )
        self._dataloaders["rollout"] = torch.utils.data.DataLoader(
            self._datasets["rollout"],
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self._dataloaders["val"] = torch.utils.data.DataLoader(
            self._datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=config.get("num_workers", 1),
            collate_fn=collate_fn,
        )
        self._warmup_steps = config.get("warmup_steps")
        self._logger = logger
        # Optimization params
        self._lr_init = config["lr_init"]
        self._lr_decay_rate = config["lr_decay_rate"]
        self._lr_decay_steps = int(config["lr_decay_steps"])
        self._loss_report_step = int(config["loss_report_step"])
        self._log_grad_step = int(config.get("log_grad_step", 2000))
        self._save_model_step = int(config["save_model_step"])
        self._eval_step = int(config["eval_step"])
        self._optimizer = torch.optim.Adam(self._sim.parameters(), lr=self._lr_init)
        self._clip_norm = config.get("clip_norm")
        self._stop_step = int(config.get("training_steps"))

        self._global_step = 0
        self._warm_up = True
        # Evaluation params
        self._rollout_steps = config["rollout_steps"]
        self._run_validate = config["run_validate"]
        self._num_eval_rollout = config.get("num_eval_rollout", 10)
        self._log_grad = config.get("log_grad", False)
        self._save_video = config.get("save_video", False)
        self._video_width = config.get("video_width", 320)
        self._video_height = config.get("video_height", 240)
        # Figure for grad plot
        self._fig = init_fig()

        # Load existing model and continue training
        self.__resume_training()

        optimizer_to(self._optimizer, self._device)

        # Weights&Biases configuration
        wandb_run_id = None
        if os.path.isfile(os.path.join(self._logger.log_folder, RUN_FILE)):
            with open(os.path.join(self._logger.log_folder, RUN_FILE), "r") as f:
                wandb_run_id = json.load(f)["run_id"]
            self._logger.print(
                f"Found run ID '{wandb_run_id}' in {self._logger.log_folder}"
            )

        device_name = (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if device.type == "cuda"
            else "cpu"
        )
        wandb_config = {
            "batch_size": config["batch_size"],
            "model": "FIGNet",
            "dataset": os.path.basename(os.path.normpath(config["data_path"])),
            "augmented": False,
            "channels": config["latent_dim"],
            "layers": config["message_passing_steps"],
            "mlp_layers": config["mlp_layers"],
            "activation_func": "relu",
            "epochs": int(config["training_steps"] / len(self._dataloaders["train"])),
            "loss": "MSE",
            "normalize": "fignet",
            "optimizer": "adam",
            "lr": config["lr_init"],
            "train_batches": len(self._dataloaders["train"]),
            "val_items": len(self._datasets["val"]),
            "workers": config["num_workers"],
            "connectivity_radius": config["data_config"]["connectivity_radius"],
            "noise_std": config["data_config"]["noise_std"],
            "gpu": device_name,
            "log_folder": self._logger.log_folder,
            "seed": config["seed"],
        }
        if wandb_run_id is not None:
            self._logger.print(f"Resuming W&B run '{wandb_run_id}'", level="debug")
            # Resume the existing run
            run = wandb.init(
                project="physics-learning-tdl",
                entity="amaury-wei",
                id=wandb_run_id,
                resume="allow",
                mode="online" if config["online"] else "disabled",
            )
        else:
            # Create a new W&B run
            run = wandb.init(
                project="physics-learning-tdl",
                entity="amaury-wei",
                job_type="train",
                config=wandb_config,
                mode="online" if config["online"] else "disabled",
            )
            wandb_run_id = run.id
            with open(os.path.join(self._logger.log_folder, RUN_FILE), "w") as f:
                json.dump({"run_id": wandb_run_id}, f)
            self._logger.print(f"Created W&B run '{wandb_run_id}'", level="debug")

        wandb.define_metric("batch")  # Create a custom X-axis
        wandb.define_metric("train/*", summary="last", step_metric="batch")
        wandb.define_metric("val/*", summary="last", step_metric="batch")
    
    def __resume_training(self) -> None:
        checkpoints_folder = os.path.join(self._logger.log_folder, "models")

        # List the current checkpoints in the run directory (if any)
        checkpoint_prefix = "weights_itr_"
        checkpoints = [
            f for f in os.listdir(checkpoints_folder) if f.startswith(checkpoint_prefix)
        ]
        checkpoints = sorted(checkpoints, key=lambda f: int(f[:-5].split("_")[-1]))

        # Grab the latest checkpoint if possible
        if len(checkpoints) > 0:
            checkpoint = checkpoints[-1]
            train_state = "train_state_itr_" + checkpoint.split("_")[-1]
            self._logger.print(f"Found checkpoint '{checkpoint}'; resuming training...")

            self._sim.load(os.path.join(checkpoints_folder, checkpoint))
            self._warm_up = False
            state_dict = torch.load(
                os.path.join(checkpoints_folder, train_state),
                map_location=self._device,
                weights_only=False,
            )
            self._optimizer.load_state_dict(state_dict["optimizer_state"])
            self._global_step = state_dict["global_train_state"]["step"]
        else:
            self._logger.print("No checkpoint found; starting training from scratch")

    def fill_normalization_buffer(self) -> None:
        """Run some steps to fill the buffer to calculate normalization
        stats"""
        warmup_steps = self._warmup_steps
        self._sim.train()
        pbar = tqdm.tqdm(range(warmup_steps), desc="Filling normalization buffer")
        for i, sample in enumerate(self._dataloaders["train"]):
            m_mask = (
                sample.node_sets[NodeType.MESH].kinematic == KinematicType.DYNAMIC
            ).squeeze()
            sample = ToTensor(self._device)(sample)

            self._sim._encoder_preprocessor(sample)
            self._sim.normalize_accelerations(
                sample.node_sets[NodeType.MESH].target[m_mask]
            )
            self.check_normalization_stats()
            pbar.update(1)
            if i > warmup_steps:
                break

    def check_normalization_stats(self):
        """Make sure the stats have no nan"""
        if torch.isnan(self._sim._node_normalizer._std_with_epsilon()).all().item():
            raise RuntimeError("Nan normalization stats")
        if torch.isnan(self._sim._output_normalizer._std_with_epsilon()).all().item():
            raise RuntimeError("Nan normalization stats")

    def train(self):
        """Run the training loop"""
        self._sim.to(self._device)
        step = self._global_step
        if self._warm_up:
            self.fill_normalization_buffer()
        else:
            self.check_normalization_stats()
        remaining_steps = len(self._dataloaders["train"])
        if self._stop_step is not None:
            remaining_steps = max(remaining_steps, self._stop_step)
        pbar = tqdm.tqdm(range(remaining_steps), desc="Training")
        pbar.update(step)
        stop_training = False
        epoch_i = int(self._global_step / len(self._dataloaders["train"]))
        while not stop_training:
            self._logger.print(f"Training on epoch {epoch_i}")
            for batch_idx, sample in enumerate(self._dataloaders["train"]):

                self._optimizer.zero_grad()
                self._sim.train()

                loss = self.cal_loss(sample)

                loss.backward()
                if self._clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self._sim.parameters(), self._clip_norm
                    )
                self._optimizer.step()

                lr_new = (
                    self._lr_init * self._lr_decay_rate ** (step / self._lr_decay_steps)
                    + 1e-6
                )
                for param in self._optimizer.param_groups:
                    param["lr"] = lr_new

                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr_new,
                        "batch": epoch_i * len(self._dataloaders["train"]) + batch_idx,
                        "epoch": epoch_i,
                    }
                )

                results = {
                    "loss": {
                        "total": loss.item(),
                    }
                }
                self.log_results(results, step)
                if self._log_grad:
                    self.log_gradients_in_model(step)
                if self._run_validate:
                    self.validate(step)
                    self.rollout_trajectories(step)

                pbar.update(1)
                if step + 1 >= remaining_steps:
                    stop_training = True
                    self._logger.print("Finished Training")
                    break
                step += 1
            epoch_i += 1

        pbar.close()
        self.save_model(step, hard_save=True)

    def cal_loss(self, sample):
        """Calculate loss"""
        non_kinematic_mask = (
            sample.node_sets[NodeType.MESH].kinematic == KinematicType.DYNAMIC
        ).to(self._device)
        sample = ToTensor(self._device)(sample)
        pred_acc, _ = self._sim.predict_accelerations(sample)
        target_acc = self._sim.normalize_accelerations(
            sample.node_sets[NodeType.MESH].target
        )
        loss = torch.nn.functional.mse_loss(pred_acc, target_acc, reduction="none")
        loss = loss.sum(dim=-1)
        num_non_kinematic = non_kinematic_mask.sum()
        loss = loss * non_kinematic_mask.squeeze()
        loss = loss.sum() / num_non_kinematic
        return loss

    def save_model(self, step: int, hard_save: bool = False):
        """Save weights and train states"""
        if (step % self._save_model_step == 0) or hard_save:
            ckpt_path = os.path.join(
                self._logger.log_folder,
                "models",
                "weights_itr_{}.ckpt".format(step),
            )
            train_state_path = os.path.join(
                self._logger.log_folder,
                "models",
                "train_state_itr_{}.ckpt".format(step),
            )
            self._sim.save(ckpt_path)
            train_state = dict(
                optimizer_state=self._optimizer.state_dict(),
                global_train_state={"step": step},
            )
            torch.save(train_state, train_state_path)

    def log_gradients_in_model(self, step):
        """Log gradient flow for debugging"""
        if step % self._log_grad_step == 0:
            plot_grad_flow(self._sim.named_parameters(), self._fig)
            self._fig.tight_layout()
            self._logger.tb.add_figure("grad_flow", self._fig, step)
            no_grad_tags = []
            for tag, value in self._sim.named_parameters():
                if value.grad is not None:
                    self._logger.tb.add_histogram(tag + "/grad", value.grad.cpu(), step)
                else:
                    no_grad_tags.append(tag)
            return no_grad_tags

    def log_results(self, results: dict, step: int):
        """Log results after each iteration"""
        self._logger.tb.add_scalar("loss/total_loss", results["loss"]["total"], step)
        if step % self._loss_report_step == 0:
            self._logger.print(f"Loss: {results['loss']['total']}.")
        self.save_model(step)

    def validate(self, step: int) -> None:
        """Compute one-timestep translational and rotational errors on the entire
        validation set.

        Args:
            step (int): Current training step
        """
        if step % self._eval_step != 0:
            return

        self._logger.print(f"Running validation after {step} steps")
        self._sim.eval()
        onestep_mse: list[float] = []
        onestep_mae: list[float] = []
        for sample in tqdm.tqdm(
            self._dataloaders["val"],
            desc="Validation",
            total=len(self._dataloaders["val"]),
        ):
            non_kinematic_mask = (
                sample.node_sets[NodeType.MESH].kinematic == KinematicType.DYNAMIC
            ).to(self._device)
            sample = ToTensor(self._device)(sample)
            pred_acc, _ = self._sim.predict_accelerations(sample)
            target_acc = self._sim.normalize_accelerations(
                sample.node_sets[NodeType.MESH].target
            )

            # Compute losses
            num_non_kinematic = non_kinematic_mask.sum()
            mse = torch.nn.functional.mse_loss(pred_acc, target_acc, reduction="none")
            mse = mse.sum(dim=-1)
            mse = mse * non_kinematic_mask.squeeze()
            mse = mse.sum() / num_non_kinematic
            onestep_mse.append(mse.item())

            mae = torch.nn.functional.l1_loss(pred_acc, target_acc, reduction="none")
            mae = mae.sum(dim=-1)
            mae = mae * non_kinematic_mask.squeeze()
            mae = mae.sum() / num_non_kinematic
            onestep_mae.append(mae.item())

        self._logger.tb.add_scalar("val/mse", np.mean(onestep_mse), step)
        self._logger.tb.add_scalar("val/mse_std", np.std(onestep_mse), step)
        self._logger.tb.add_scalar("val/mae", np.mean(onestep_mae), step)
        self._logger.tb.add_scalar("val/mae_std", np.std(onestep_mae), step)
        wandb.log(
            {
                "val/mse": np.mean(onestep_mse),
                "val/mse_std": np.std(onestep_mse),
                "val/mae": np.mean(onestep_mae),
                "val/mae_std": np.std(onestep_mae),
                "batch": step,
            }
        )

    def rollout_trajectories(self, step: int):
        """Compute a few multi-step rollouts and calculate translational and rotational
        errors on the last step.

        Args:
            step (int): Current training step
        """
        if step % self._eval_step != 0 and self._num_eval_rollout is not None:
            return

        input_seq_length = self._input_seq_length
        self._sim.eval()
        rollout_t_errors = []
        rollout_r_errors = []
        for i, data in tqdm.tqdm(
            enumerate(self._dataloaders["rollout"]),
            desc="Validation rollouts",
            total=self._num_eval_rollout,
        ):
            if i >= self._num_eval_rollout:
                break
            # Get initial states
            traj = data[0]
            mujoco_xml = data[1]
            scene_config = data[2]
            init_poses = traj["pose_seq"][0:input_seq_length, ...]
            gt_poses = to_numpy(traj["pose_seq"])
            gt_poses = np.delete(gt_poses, 0, axis=0)
            obj_ids = traj["obj_ids"]
            for k, v in obj_ids.items():
                obj_ids[k] = v.cpu().item()
            pred_traj = []
            try:
                pred_traj = rollout(
                    sim=self._sim,
                    init_obj_poses=init_poses,
                    obj_ids=obj_ids,
                    scene=Scene(scene_config),
                    device=self._device,
                    nsteps=self._rollout_steps,
                    quiet=True,
                )
            except Exception as e:
                self._logger.print(
                    f"Failed to evaluate rollout {i} with exception: {e}",
                    level="warn",
                )
                return

            # Calculate rollout errors
            pred_pose = pred_traj[-1, ...]
            gt_pose = gt_poses[-1, ...]
            error_t = (gt_pose[:, :3] - pred_pose[:, :3]) ** 2
            error_t = np.mean(error_t.sum(axis=-1))
            error_t = np.sqrt(error_t)  # RMSE = Root Mean Square Error

            pred_pose[:, 4:7] *= -1  # Compute quaternion conjugate
            error_r = orientation_error_deg(gt_pose[:, 3:], pred_pose[:, 3:])
            error_r = np.mean(error_r)
            rollout_t_errors.append(error_t)
            rollout_r_errors.append(error_r)

            # Synthesize video using predicted rollout (only validation item 0)
            if i == 0 and self._dataset_type == "Mujoco" and self._save_video:
                try:
                    screens_pred = visualize_trajectory(
                        mujoco_xml=mujoco_xml,
                        traj=pred_traj,
                        obj_ids=obj_ids,
                        height=self._video_height,
                        width=self._video_width,
                        off_screen=True,
                    )
                    screens_gt = visualize_trajectory(
                        mujoco_xml=mujoco_xml,
                        traj=gt_poses,
                        obj_ids=obj_ids,
                        height=self._video_height,
                        width=self._video_width,
                        off_screen=True,
                    )
                    screens_pred = screens_pred.transpose(0, 3, 1, 2)
                    screens_gt = screens_gt.transpose(0, 3, 1, 2)
                    self._logger.tb.add_video(
                        "video/simulated",
                        torch.from_numpy(screens_pred[None, :]),
                        step,
                        fps=60,
                    )
                    self._logger.tb.add_video(
                        "video/ground_truth",
                        torch.from_numpy(screens_gt[None, :]),
                        step,
                        fps=60,
                    )
                except Exception as e:
                    self._logger.print(e, level="warn")

        self._logger.tb.add_scalar(
            "val/rollout_rmse_trsl",
            np.mean(rollout_t_errors),
            step,
        )
        self._logger.tb.add_scalar(
            "val/rollout_rmse_rot",
            np.mean(rollout_r_errors),
            step,
        )
        wandb.log(
            {
                "val/rollout_rmse_trsl": np.mean(rollout_t_errors),
                "val/rollout_rmse_rot": np.mean(rollout_r_errors),
                "batch": step,
            }
        )


def create_trainer(config_file: str, logging_folder: str = "", seed: int | None = None):
    """Create trainer object"""
    with open(os.path.join(os.getcwd(), config_file)) as f:
        if config_file.endswith("yaml"):
            config = yaml.safe_load(f)
        elif config_file.endswith("json"):
            print("Warning: json config file will not be supported soon")
            import json

            config = json.load(f)
        else:
            raise TypeError("Unsupported config file type")

    if logging_folder is not None and logging_folder != "":
        config["logging_folder"] = logging_folder
        print(f"WARN: Overriding logging folder: {logging_folder}")

    logger = Logger(config)
    if torch.cuda.is_available() and config.get("use_cuda", True):
        device = torch.device("cuda")
        logger.print(f"Using GPU {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.print(f"Using MPS backend")
    else:
        device = torch.device("cpu")
        logger.print("Using CPU", level="warn")

    if seed is not None:
        config["seed"] = seed
        logger.print(f"Overriding seed: {seed}")

    sim = LearnedSimulator(
        mesh_dimensions=3,
        latent_dim=config.get("latent_dim", 128),
        nmessage_passing_steps=config.get("message_passing_steps", 10),
        nmlp_layers=config.get("mlp_layers", 2),
        input_seq_length=config["data_config"]["input_seq_length"],
        mlp_hidden_dim=config.get("latent_dim", 128),
        device=device,
        leave_out_mm=config.get("leave_out_mm", False),
    )
    sim = torch.compile(sim)
    trainer = Trainer(sim=sim, logger=logger, config=config, device=device)

    return trainer
