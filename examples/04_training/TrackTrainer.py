import os
from typing import Any, Dict, List, Optional
from datetime import datetime

import torch
from torch.optim.adam import Adam
from pytorch_lightning.loggers import WandbLogger

from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel, Model
from graphnet.models.detector.pone import PONE
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset, ParquetDataset
from graphnet.utilities.config import ModelConfig
from graphnet.models.graphs.nodes import PercentileClusters

torch.multiprocessing.set_start_method('spawn', force=True)

# Constants
features = FEATURES.PONE
truth = TRUTH.PONE

def get_latest_model_dir(save_path: str) -> Optional[str]:
    """Find the most recent model directory in save_path."""
    subdirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]
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
    logger = Logger()
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

    graph_definition = KNNGraph(detector=PONE(),
        node_definition=PercentileClusters(cluster_on=['module_location_x', 'module_location_y', 'module_location_z'],
        percentiles=[0, 10, 50, 90, 100],
        input_feature_names=features,))

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
        train_dataloader_kwargs={"batch_size": config["batch_size"], "num_workers": config["num_workers"]},
        validation_dataloader_kwargs={"batch_size": config["batch_size"], "num_workers": config["num_workers"]},
        test_dataloader_kwargs={"batch_size": config["batch_size"], "num_workers": config["num_workers"]},
    )

    training_dataloader, validation_dataloader = dm.train_dataloader, dm.val_dataloader

    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )

    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=BinaryCrossEntropyLoss(),
        transform_prediction_and_target=lambda x: x,
        transform_inference=lambda x: x,
    )

    latest_model_dir = get_latest_model_dir(save_path)

    if latest_model_dir:
        model_config = ModelConfig.load(f"{latest_model_dir}/model_config.yml")
        model = Model.from_config(model_config, trust=True)
        model.load_state_dict(torch.load(f"{latest_model_dir}/state_dict.pth"))
    else:
        model = StandardModel(
            graph_definition=graph_definition,
            backbone=backbone,
            tasks=[task],
            optimizer_class=Adam,
            optimizer_kwargs={"lr": 1e-03, "eps": 1e-05},
            scheduler_class=PiecewiseLinearLR,
            scheduler_kwargs={
                "milestones": [0, len(training_dataloader) / 2, len(training_dataloader) * max_epochs],
                "factors": [1e-2, 0.5, 1e-02],
            },
            scheduler_config={"interval": "step"},
        )

    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        **config["fit"],
    )

    new_save_path = os.path.join(save_path, datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(new_save_path, exist_ok=True)

    model.save(f"{new_save_path}/model.pth")
    torch.save(model.state_dict(), f"{new_save_path}/state_dict.pth")
    model.save_config(f"{new_save_path}/model_config.yml")
    

if __name__ == "__main__":
    DATASET_PATH = "/raven/ptmp/arego/R1T4K_100_parquet"
    SAVE_PATH = "/raven/ptmp/arego/Class/TrackTrainer"
    PULSEMAP = "hits"
    TARGET = "type"
    TRUTH_TABLE = "records"
    GPUS = [0]
    MAX_EPOCHS = 20
    EARLY_STOPPING_PATIENCE = 5
    BATCH_SIZE = 128
    NUM_WORKERS = 12
    WANDB = False

    main(
        DATASET_PATH,
        SAVE_PATH,
        PULSEMAP,
        TARGET,
        TRUTH_TABLE,
        GPUS,
        MAX_EPOCHS,
        EARLY_STOPPING_PATIENCE,
        BATCH_SIZE,
        NUM_WORKERS,
        WANDB,
    )