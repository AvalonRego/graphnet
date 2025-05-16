"""Example of training Model.

This example is based on Icemix solution proposed in
https://github.com/DrHB/icecube-2nd-place.git
(2nd place solution).
"""
import os
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch
from torch.optim.adam import Adam
from pytorch_lightning.loggers import WandbLogger

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel, Model
from graphnet.models.detector.pone import PONE  # PONE detector
from graphnet.models.gnn import DeepIce
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import IceMixNodes  # IceMix
from graphnet.models.task.classification import BinaryClassificationTask
#from graphnet.training.callbacks import ReduceLROnPlateau
from torch.optim.lr_scheduler import ReduceLROnPlateau
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset, ParquetDataset
from graphnet.utilities.config import ModelConfig

torch.multiprocessing.set_start_method("spawn", force=True)

# Constants
features = FEATURES.PONE  # PONE features
truth = TRUTH.PONE  # PONE truth


def get_latest_model_dir(save_path: str) -> Optional[str]:
    """Find the most recent model directory in save_path."""
    subdirs = [
        d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))
    ]
    if not subdirs:
        return None
    latest_dir = max(subdirs, key=lambda d: datetime.strptime(d, "%Y%m%d_%H%M%S"))
    return os.path.join(save_path, latest_dir)


def main(
    path: str,
    save_path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    max_epochs: int,
    early_stopping_patience: int,
    batch_size: int,
    num_workers: int,
    wandb: bool = False,
) -> None:
    """Train DeepIce + IceMix with binary classification task for PONE detector."""
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        wandb_dir = "./wandb/"
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_logger = WandbLogger(
            project="example-script",
            entity="graphnet-team",
            save_dir=wandb_dir,
            log_model=True,
        )

    logger.info(f"features: {features}")
    logger.info(f"truth: {truth}")

    # Configuration
    config: Dict[str, Any] = {
        "path": path,
        "save_path": save_path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
                    "gpus": gpus,
                    "max_epochs": max_epochs,
                    "distribution_strategy": "ddp_find_unused_parameters_true",
        },
        "dataset_reference": SQLiteDataset if path.endswith(".db") else ParquetDataset,
    }

    if wandb:
        wandb_logger.experiment.config.update(config)

    # Define graph structure with IceMixNodes and PONE
    graph_definition = KNNGraph(
        detector=PONE(),
        node_definition=IceMixNodes(
            input_feature_names=features,
            max_pulses=128,
            z_name="module_location_z",
            hlc_name=None,
            add_ice_properties=False,
        ),
        input_feature_names=features,
        columns=[0, 1, 2, 3],
    )

    # Load dataset
    dm = GraphNeTDataModule(
        dataset_reference=config["dataset_reference"],
        dataset_args={
            "truth": truth,
            "truth_table": truth_table,
            "features": features,
            "graph_definition": graph_definition,
            "pulsemaps": [config["pulsemap"]],
            "path": config["path"],
        },
        train_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
        validation_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
        test_dataloader_kwargs={
            "batch_size": config["batch_size"],
            "num_workers": config["num_workers"],
        },
    )

    training_dataloader= dm.train_dataloader
    validation_dataloader = dm.val_dataloader

    # Define DeepIce backbone
    backbone = DeepIce(
        hidden_dim=768,
        seq_length=192,
        depth=12,
        head_size=64,
        n_rel=4,
        scaled_emb=True,
        include_dynedge=True,
        dynedge_args={
            "nb_inputs": graph_definition._node_definition.n_features,
            "nb_neighbours": 9,
            "post_processing_layer_sizes": [336, 384],
            "activation_layer": "gelu",
            "add_norm_layer": True,
            "skip_readout": True,
        },
        n_features=len(features),
    )

    # Define binary classification task
    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=BinaryCrossEntropyLoss(),
        transform_prediction_and_target=lambda x: x,
        transform_inference=lambda x: x,
    )

    # Load latest saved model if available
    latest_model_dir = get_latest_model_dir(save_path)

    if latest_model_dir:
        logger.info(f"Loading model from {latest_model_dir}")
        model_config = ModelConfig.load(f"{latest_model_dir}/model_config.yml")
        model = Model.from_config(model_config, trust=True)
        model.load_state_dict(torch.load(f"{latest_model_dir}/state_dict.pth"))
    else:
        model = StandardModel(
            graph_definition=graph_definition,
            backbone=backbone,
            tasks=[task],
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 1e-03, "eps": 1e-08},
            scheduler_class=ReduceLROnPlateau,
            scheduler_kwargs={"patience": 6, },
            scheduler_config={
            "interval": "epoch", "monitor": "val_loss",
        },
        )

    # Train the model
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        **config["fit"],
    )

    # Save the model
    new_save_path = os.path.join(save_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(new_save_path, exist_ok=True)

    model.save(f"{new_save_path}/model.pth")
    torch.save(model.state_dict(), f"{new_save_path}/state_dict.pth")
    model.save_config(f"{new_save_path}/model_config.yml")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = ArgumentParser()

    parser.add_argument("--path", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--pulsemap", default="hits")
    parser.add_argument("--target", default="duration")
    parser.add_argument("--truth-table", default="records")

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 10),
        ("early-stopping-patience", 5),
        ("batch-size", 128),
        "num-workers",
    )

    parser.add_argument("--wandb", action="store_true")

    args, unknown = parser.parse_known_args()

    main(
        args.path,
        args.save_path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.max_epochs,
        args.early_stopping_patience,
        args.batch_size,
        args.num_workers,
        args.wandb,
    )
