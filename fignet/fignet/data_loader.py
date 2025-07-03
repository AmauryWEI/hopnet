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
import pickle
from dataclasses import fields
from pathlib import Path
from typing import Callable, List

import numpy as np
import torch
from torch.utils.data import Dataset

from fignet.scene import Scene
from fignet.types import EdgeType, Graph, NodeType
from fignet.utils import dataclass_to_tensor, dict_to_tensor, download_movi_mesh


def collate_fn(batch: List[Graph]):
    """Merge batch of graphs into one graph"""
    if len(batch) == 1:
        return batch[0]
    else:
        batch_graph = batch.pop(0)
        m_node_offset = batch_graph.node_sets[NodeType.MESH].kinematic.shape[0]
        o_node_offset = batch_graph.node_sets[NodeType.OBJECT].kinematic.shape[0]
        for graph in batch:
            for node_typ in graph.node_sets.keys():
                for field in fields(graph.node_sets[node_typ]):
                    if field.name == "position":
                        cat_dim = 1
                    else:
                        cat_dim = 0
                    setattr(
                        batch_graph.node_sets[node_typ],
                        field.name,
                        torch.cat(
                            [
                                getattr(batch_graph.node_sets[node_typ], field.name),
                                getattr(graph.node_sets[node_typ], field.name),
                            ],
                            dim=cat_dim,
                        ),
                    )
            for edge_typ in graph.edge_sets.keys():
                if edge_typ == EdgeType.MESH_MESH or edge_typ == EdgeType.FACE_FACE:
                    graph.edge_sets[edge_typ].index += m_node_offset
                elif edge_typ == EdgeType.OBJ_MESH:
                    graph.edge_sets[edge_typ].index[0, :] += o_node_offset
                    graph.edge_sets[edge_typ].index[1, :] += m_node_offset
                elif edge_typ == EdgeType.MESH_OBJ:
                    graph.edge_sets[edge_typ].index[0, :] += m_node_offset
                    graph.edge_sets[edge_typ].index[1, :] += o_node_offset
                else:
                    raise TypeError(f"Unknown edge type {edge_typ}")
                # Concatenate
                batch_graph.edge_sets[edge_typ].index = torch.cat(
                    [
                        batch_graph.edge_sets[edge_typ].index,
                        graph.edge_sets[edge_typ].index,
                    ],
                    dim=1,
                )
                batch_graph.edge_sets[edge_typ].attribute = torch.cat(
                    [
                        batch_graph.edge_sets[edge_typ].attribute,
                        graph.edge_sets[edge_typ].attribute,
                    ],
                    dim=0,
                )
            m_node_offset += graph.node_sets[NodeType.MESH].kinematic.shape[0]
            o_node_offset += graph.node_sets[NodeType.OBJECT].kinematic.shape[0]

        return batch_graph


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, device=None):
        self.device = device

    def __call__(self, sample):
        # convert numpy arrays to pytorch tensors
        if isinstance(sample, dict):
            return dict_to_tensor(sample, self.device)
        elif isinstance(sample, Graph):
            return dataclass_to_tensor(sample, self.device)
        else:
            raise TypeError(f"Cannot convert {type(sample)} to tensor.")


class MujocoDataset(Dataset):
    """Load Mujoco dataset"""

    def __init__(
        self,
        path: str,
        input_sequence_length: int,
        split: str,
        mode: str,
        config: dict | None = None,
        transform=None,
    ):
        # If raw data is given, need to calculate graph connectivity on the fly
        if os.path.isfile(path):
            self._load_raw_data = True

            self._data = list(np.load(path, allow_pickle=True).values())[0]
            self._dimension = self._data[0]["pos"].shape[-1]
            self._target_length = 1
            self._input_sequence_length = input_sequence_length

            self._data_lengths = [
                x["pos"].shape[0] - input_sequence_length - self._target_length
                for x in self._data
            ]
            self._length = sum(self._data_lengths)

            # pre-compute cumulative lengths
            # to allow fast indexing in __getitem__
            self._precompute_cumlengths = [
                sum(self._data_lengths[:x])
                for x in range(1, len(self._data_lengths) + 1)
            ]
            self._precompute_cumlengths = np.array(
                self._precompute_cumlengths, dtype=int
            )
        # Directly load pre-calculated graphs and save time
        elif os.path.isdir(path):
            self._load_raw_data = False

            self._graph_ext = "pkl"
            self._file_list = [
                os.path.join(path, f)
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.endswith(self._graph_ext)
            ]
            self._file_list.sort(key=lambda f: int(Path(f).stem.split("_")[1]))
            self._length = len(self._file_list)
        else:
            raise FileNotFoundError(f"{path} not found")

        self._transform = transform
        self._mode = mode
        if config is not None:
            self._config = config
        else:
            self._config = {}

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._mode == "sample":
            return self._get_sample(idx)
        elif self._mode == "trajectory":
            return self._get_trajectory(idx)

    def _load_graph(self, graph_file):
        try:
            with open(graph_file, "rb") as f:
                sample_dict = pickle.load(f)
                graph = Graph()
                graph.from_dict(sample_dict)
            return graph

        except FileNotFoundError as e:
            print(e)
            return None

    def _get_sample(self, idx):
        """Sample one step"""
        if self._load_raw_data:
            trajectory_idx = np.searchsorted(
                self._precompute_cumlengths - 1, idx, side="left"
            )
            # Compute index of pick along time-dimension of trajectory.
            start_of_selected_trajectory = (
                self._precompute_cumlengths[trajectory_idx - 1]
                if trajectory_idx != 0
                else 0
            )
            time_idx = self._input_sequence_length + (
                idx - start_of_selected_trajectory
            )

            start = time_idx - self._input_sequence_length
            end = time_idx
            obj_ids = dict(self._data[trajectory_idx]["obj_id"].item())
            positions = self._data[trajectory_idx]["pos"][
                start:end
            ]  # (seq_len, n_obj, 3) input sequence
            quats = self._data[trajectory_idx]["quat"][
                start:end
            ]  # (seq_len, n_obj, 4) input sequence
            target_posisitons = self._data[trajectory_idx]["pos"][time_idx]
            target_quats = self._data[trajectory_idx]["quat"][time_idx]
            poses = np.concatenate([positions, quats], axis=-1)
            target_poses = np.concatenate([target_posisitons, target_quats], axis=-1)

            scene_config = dict(self._data[trajectory_idx]["meta_data"].item())

            connectivity_radius = self._config.get("connectivity_radius")
            if connectivity_radius is not None:
                scene_config.update({"connectivity_radius": connectivity_radius})

            noise_std = self._config.get("noise_std")
            if noise_std is not None:
                scene_config.update({"noise_std": noise_std})

            scn = Scene(scene_config)
            scn.synchronize_states(
                obj_poses=poses,
                obj_ids=obj_ids,
            )
            graph = scn.to_graph(
                target_poses=target_poses,
                obj_ids=obj_ids,
                noise=True,
            )

            if self._transform is not None:
                graph = self._transform(graph)
            return graph
        else:
            if os.path.exists(self._file_list[idx]):
                return self._transform(self._load_graph(self._file_list[idx]))
            else:
                raise FileNotFoundError

    def _get_trajectory(self, idx):
        """Sample continuous steps"""
        trajectory_idx = np.searchsorted(
            self._precompute_cumlengths - 1, idx, side="left"
        )
        start_of_selected_trajectory = (
            self._precompute_cumlengths[trajectory_idx - 1]
            if trajectory_idx != 0
            else 0
        )
        time_idx = self._input_sequence_length + (idx - start_of_selected_trajectory)

        start = time_idx - self._input_sequence_length
        end = time_idx
        obj_ids = self._data[trajectory_idx]["obj_id"]
        obj_ids = dict(obj_ids.item())
        mujoco_xml = str(self._data[trajectory_idx]["mujoco_xml"])
        scene_config = dict(self._data[trajectory_idx]["meta_data"].item())
        scene_config["experiment_id"] = idx

        try:
            connectivity_radius = self._config["connectivity_radius"]
            scene_config.update({"connectivity_radius": connectivity_radius})
        except KeyError:
            pass
        positions = self._data[trajectory_idx]["pos"][
            start:end
        ]  # (seq_len, n_obj, 3) input sequence
        quats = self._data[trajectory_idx]["quat"][
            start:end
        ]  # (seq_len, n_obj, 4) input sequence
        pose_seq = np.concatenate([positions, quats], axis=-1)

        traj = {"obj_ids": obj_ids, "pose_seq": pose_seq}
        if self._transform is not None:
            traj = self._transform(traj)
        return traj, mujoco_xml, scene_config


class MoviDataset(Dataset):
    # Dataset configurations
    RATIO_TRAIN: float = 0.8
    RATIO_VAL: float = 0.1
    RATIO_TEST: float = 0.1
    METADATA_FILENAME: str = "metadata.json"
    GRAPHS_DIRECTORY: str = "graphs"
    MESHES_LOCATION: str = os.path.join(
        Path(__file__).parent.resolve(), "../../data/objects"
    )

    def __init__(
        self,
        dataset_path: str,
        split: str,
        input_sequence_length: int,
        mode: str,
        config: dict | None = None,
        transform: Callable | None = None,
    ):
        self._dataset_path = dataset_path
        self._transform = transform
        self._mode = mode
        self._config: dict = config if config is not None else {}

        # Default values to match MujocoDataset
        self._dimension = 3  # (x, y, z)
        self._target_length = 1  # Predict one future timestep
        self._input_sequence_length = input_sequence_length  # Horizon + 1

        # Actual dataset configuration
        self.__experiments_paths = []
        self.__read_experiment_ids_from_disk(dataset_path)
        self.__select_split(split)

        # Find the dimensions of the trajectories, dataset size, etc...
        self.__read_dataset_dimensions()

        super().__init__()

    @property
    def graphs_directory(self) -> str:
        return os.path.join(self._dataset_path, self.GRAPHS_DIRECTORY)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        if self._mode == "sample":
            return self._get_sample(idx)
        elif self._mode == "preprocess":
            return self._get_sample(idx, skip_loading=True)
        elif self._mode == "trajectory":
            return self._get_trajectory(idx)

    def __read_experiment_ids_from_disk(self, dataset_dir: str) -> None:
        dataset_dir = os.path.abspath(dataset_dir)
        if not os.path.isdir(dataset_dir):
            raise RuntimeError(f"Dataset directory {dataset_dir} does not exist.")

        experiments_dir = [f.name for f in os.scandir(dataset_dir) if f.is_dir()]
        try:
            # Experiments with integers longer than 4 are augmented dataset (train only)
            experiments = set([int(f) for f in experiments_dir if len(f) <= 4])
        except Exception as e:
            print(f"ERROR: Dataset directory contains non-digit experiment IDs")
            raise e

        self.__experiments_paths = [
            os.path.join(dataset_dir, str(exp))
            for exp in experiments
            if os.path.isfile(
                os.path.join(dataset_dir, str(exp), self.METADATA_FILENAME)
            )
        ]

        if not len(self.__experiments_paths) > 0:
            raise RuntimeError(f"Empty dataset: {dataset_dir}")

    def __select_split(self, split: str) -> None:
        # Compute which experiments should be used for training, validation, and testing
        rng = np.random.default_rng(seed=0)
        all_indices = np.arange(len(self.__experiments_paths)).tolist()
        train_indices = rng.choice(
            all_indices,
            size=int(np.ceil(self.RATIO_TRAIN * len(self.__experiments_paths))),
            replace=False,
        )
        remaining_indices = list(set(all_indices) - set(train_indices))
        val_indices = rng.choice(
            remaining_indices,
            size=int(np.ceil(self.RATIO_VAL * len(self.__experiments_paths))),
            replace=False,
        )
        test_indices: list[int] = list(set(remaining_indices) - set(val_indices))

        # Keep only certain indices, based on the split
        if split == "train":
            self.__experiments_paths = [
                self.__experiments_paths[idx] for idx in train_indices
            ]
        elif split == "val":
            self.__experiments_paths = [
                self.__experiments_paths[idx] for idx in val_indices
            ]
        elif split == "test":
            self.__experiments_paths = [
                self.__experiments_paths[idx] for idx in test_indices
            ]
        elif split == "all":
            pass
        else:
            raise RuntimeError(f"ERROR: Unknown data split {split}")

    def __read_dataset_dimensions(self) -> None:
        """Find out all the key dimensions of the dataset (trajectories length, ...)"""
        # Load the first item of the dataset and figure out the dimensions
        metadata = os.path.join(self.__experiments_paths[0], self.METADATA_FILENAME)
        with open(metadata) as f:
            metadata = json.load(f)
        trajectory_length = len(metadata["instances"][0]["positions"])

        # Total length of the dataset
        if self._mode == "sample" or self._mode == "preprocess":
            # Usable duration of each trajectory
            trainable_timesteps = (
                trajectory_length - self._input_sequence_length - self._target_length
            )
            self._length = len(self.__experiments_paths) * trainable_timesteps
        elif self._mode == "trajectory":
            trainable_timesteps = 1
            self._length = len(self.__experiments_paths)
        else:
            raise RuntimeError(f"ERROR: Unknown dataset mode {self._mode}")

        # Cumulative lengths for fast indexing in __getitem__()
        self._precompute_cumlengths = trainable_timesteps * np.arange(
            1, len(self.__experiments_paths) + 1
        )

    def __cfg_from_metadata(self, metadata: dict) -> tuple[dict, dict, np.ndarray]:
        """Convert a MOVi metadata.json file into a Mujoco configuration"""
        num_timesteps = np.array(metadata["instances"][0]["positions"]).shape[0]
        num_objects = len(metadata["instances"])
        object_poses = np.empty((num_timesteps, num_objects, 3 + 4))
        object_ids: dict = {}

        # Hardcode floor information (taken from HOPNet dataset creation)
        data = {
            "env": {
                "floor": {
                    "type": "box",
                    "extents": [20, 20, 0.5],
                    "properties": {
                        "restitution": metadata["flags"]["floor_restitution"],
                        "friction": 3 * [metadata["flags"]["floor_friction"]],
                        "mass": 0.0,
                    },
                    "initial_pose": [0, 0, -0.25],  # z=-0.25 to match its thickness
                },
            },
            "objects": {},
        }

        # Add the information for each object
        for object_idx, object in enumerate(metadata["instances"]):
            object_key = f"{object['asset_id']}-{object_idx}"
            object_ids[object_key] = object_idx
            download_movi_mesh(object["asset_id"], self.MESHES_LOCATION)
            data["objects"][object_key] = {
                "mesh": os.path.join(
                    self.MESHES_LOCATION, object["asset_id"], "collision_geometry.obj"
                ),
                "properties": {
                    "restitution": object["restitution"],
                    "friction": 3 * [object["friction"]],
                    "mass": object["mass"],
                },
                "size": object["size"] if "size" in object else object["scale"],
            }
            object_poses[:, object_idx, :3] = object["positions"]
            object_poses[:, object_idx, 3:] = object["quaternions"]

        # Parameters taken from training configuration
        connectivity_radius = self._config.get("connectivity_radius")
        if connectivity_radius is not None:
            data.update({"connectivity_radius": connectivity_radius})
        noise_std = self._config.get("noise_std")
        if noise_std is not None:
            data.update({"noise_std": noise_std})

        return data, object_ids, object_poses

    def _load_graph(self, graph_file: str) -> Graph:
        with open(graph_file, "rb") as f:
            sample_dict = pickle.load(f)
            graph = Graph()
            graph.from_dict(sample_dict)
        return graph

    def _get_sample(self, idx: int, skip_loading: bool = False) -> Graph | None:
        """Get a single timestep"""
        graph_path = os.path.join(
            self._dataset_path, self.GRAPHS_DIRECTORY, f"graph_{idx}.pkl"
        )

        # Check if the graph is already pre-computed
        if os.path.exists(graph_path):
            if skip_loading:
                return None
            # Load the graph directly from the disk
            graph = self._load_graph(graph_path)
            return self._transform(graph) if self._transform is not None else graph
        # Compute the required graph manually
        else:
            experiment_idx = np.searchsorted(
                self._precompute_cumlengths - 1, idx, side="left"
            )
            # Compute index of pick along time-dimension of trajectory.
            start_of_selected_trajectory = (
                self._precompute_cumlengths[experiment_idx - 1]
                if experiment_idx != 0
                else 0
            )
            time_idx = self._input_sequence_length + (
                idx - start_of_selected_trajectory
            )

            start = time_idx - self._input_sequence_length
            end = time_idx

            # Load the metadata.json and convert it to the required format
            metadata_file = os.path.join(
                self.__experiments_paths[experiment_idx], self.METADATA_FILENAME
            )
            with open(metadata_file) as f:
                metadata = json.load(f)
            scene_config, obj_ids, obj_poses = self.__cfg_from_metadata(metadata)

            # Extract the right poses and targets
            scn = Scene(scene_config)
            scn.synchronize_states(
                obj_poses=obj_poses[start:end],
                obj_ids=obj_ids,
            )
            graph = scn.to_graph(
                target_poses=obj_poses[time_idx],
                obj_ids=obj_ids,
                noise=True,
            )

            if self._transform is not None:
                graph = self._transform(graph)
            return graph

    def _get_trajectory(self, idx) -> tuple[dict, str, dict]:
        """Sample continuous steps"""
        experiment_idx = idx
        start_of_selected_trajectory = 0
        time_idx = self._input_sequence_length - start_of_selected_trajectory

        start = time_idx - self._input_sequence_length
        end = time_idx

        # Load the metadata.json and convert it to the required format
        metadata_file = os.path.join(
            self.__experiments_paths[experiment_idx], self.METADATA_FILENAME
        )
        with open(metadata_file) as f:
            metadata = json.load(f)
        scene_config, obj_ids, obj_poses = self.__cfg_from_metadata(metadata)
        scene_config["experiment_id"] = os.path.basename(self.__experiments_paths[idx])

        traj = {"obj_ids": obj_ids, "pose_seq": obj_poses[start:end]}
        if self._transform is not None:
            traj = self._transform(traj)
        return traj, "", scene_config
