import os
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch
from torch.optim.adam import Adam
from pytorch_lightning.loggers import WandbLogger

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel, Model
from graphnet.models.detector.pone import PONE  # PONE detector
from graphnet.models.gnn import RNN_TITO  # RNN-TITO backbone
from graphnet.models.graphs import KNNGraph
from graphnet.models.graphs.nodes import NodeAsDOMTimeSeries  # Time-series nodes
from graphnet.models.task.classification import BinaryClassificationTask
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
    """Train RNN-TITO with binary classification task for PONE detector."""
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
        "fit": {"gpus": gpus, "max_epochs": max_epochs},
        "dataset_reference": SQLiteDataset if path.endswith(".db") else ParquetDataset,
    }

    if wandb:
        wandb_logger.experiment.config.update(config)

    # Define graph structure with time-series nodes
    graph_definition = KNNGraph(
        detector=PONE(),
        node_definition=NodeAsDOMTimeSeries(
            keys=features,
            id_columns=features[0:3],  # Spatial ID columns
            time_column=features[-1],  # Time column
            charge_column="None",
        ),
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
            "index_column": "event_no",
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

    training_dataloader, validation_dataloader = dm.train_dataloader, dm.val_dataloader

    # Define RNN-TITO backbone
    backbone = RNN_TITO(
        nb_inputs=graph_definition.nb_outputs,
        nb_neighbours=8,
        time_series_columns=[4, 3],  # Specify time-series feature indices
        rnn_layers=2,
        rnn_hidden_size=64,
        rnn_dropout=0.5,
        features_subset=[0, 1, 2, 3],
        dyntrans_layer_sizes=[(256, 256), (256, 256), (256, 256), (256, 256)],
        post_processing_layer_sizes=[336, 256],
        readout_layer_sizes=[256, 128],
        global_pooling_schemes=["max"],
        embedding_dim=0,
        n_head=16,
        use_global_features=True,
        use_post_processing_layers=True,
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
             scheduler_kwargs={
            "patience": config["early_stopping_patience"],
        },
        scheduler_config={
            "frequency": 1,
            "monitor": "val_loss",
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
        ("max-epochs", 4),
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
