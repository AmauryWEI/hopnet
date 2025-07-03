# File:         preprocess_data_movi.py
# Date:         2024/05/28
# Description:  Create compressed graphs for training FIGNet on MOVi dataset

import argparse
import os
import pickle

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm
from yaml import safe_load

from fignet.data_loader import MoviDataset, ToTensor

torch.multiprocessing.set_sharing_strategy("file_system")


def collate_fn(batch):
    return batch


def save_graph(graph, graph_i, save_path):
    if isinstance(graph, list):
        batch_size = len(graph)
        for g_i, g in enumerate(graph):
            if g is None:  # None = graph already exists ("preproces" mode)
                continue
            i = graph_i * batch_size + g_i
            save_graph(g, i, save_path)
    else:
        graph_dict = graph.to_dict()
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        file_name = os.path.join(save_path, f"graph_{graph_i}.pkl")
        with open(file_name, "wb") as f:
            pickle.dump(graph_dict, f)


def main(args: argparse.Namespace) -> int:
    # Make sure the config file exists
    config_path = os.path.join(os.getcwd(), args.config_file)
    if not os.path.isfile(config_path):
        print(f"ERROR: Cannot find config file {config_path}")
        return 1

    # Parse the config file and log some core information
    with open(config_path) as f:
        config = safe_load(f)
    dataset_path = os.path.abspath(args.dataset_path)
    print(f"INFO: Parsing data directory {dataset_path}")

    # Adjust some config parameters
    num_workers = max(args.num_workers, 1)
    batch_size = min(2 * num_workers, 64)
    device = torch.device("cpu")

    dataset = MoviDataset(
        dataset_path=dataset_path,
        split="all",
        mode="preprocess",
        input_sequence_length=3,
        transform=T.Compose([ToTensor(device)]),
        config=config.get("data_config"),
    )
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn,
    )
    for i, sample in enumerate(tqdm(data_loader, desc="Preprocessing data")):
        save_graph(sample, i, dataset.graphs_directory)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=1)
    args = parser.parse_args()
    ret = main(args)
    exit(ret)
