
import os
from typing import Any, Dict, List, Optional

from pytorch_lightning.loggers import WandbLogger
import torch
from torch.optim.adam import Adam

#from graphnet.constants import EXAMPLE_DATA_DIR, EXAMPLE_OUTPUT_DIR
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models import StandardModel
from graphnet.models.detector.pone import PONE
from graphnet.models.gnn import DynEdge
from graphnet.models.graphs import KNNGraph
from graphnet.models.task.classification import BinaryClassificationTask
from graphnet.training.callbacks import PiecewiseLinearLR
from graphnet.training.loss_functions import BinaryCrossEntropyLoss
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datetime import datetime

from graphnet.models import Model
from graphnet.utilities.config import ModelConfig


torch.multiprocessing.set_start_method('spawn', force=True)



# Constants
features = FEATURES.PONE
truth = TRUTH.PONE


def main(
    path: str,
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
    """Run example."""
    # Construct Logger
    logger = Logger()

    # Initialise Weights & Biases (W&B) run
    if wandb:
        # Make sure W&B output directory exists
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
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "early_stopping_patience": early_stopping_patience,
        "fit": {
            "gpus": gpus,
            "max_epochs": max_epochs,
        },
        "dataset_reference": (
            SQLiteDataset if path.endswith(".db") else ParquetDataset
        ),
    }

    def get_timestamp():
        """Generate a timestamp with month, day, hour, and minute."""
        return datetime.now().strftime("%b-%d_%H-%M")

    #archive = os.path.join('/raven/ptmp/arego/Class/', "Training")
    #run_name = f"Training_{config['target']}_{get_timestamp()}"
    run_name='/raven/ptmp/arego/Class/Training/R1T4K_100_parquet/Training_type_Mar-10_12-44/'
    if wandb:
        # Log configuration to W&B
        wandb_logger.experiment.config.update(config)

    # Define graph/data representation, here the KNNGraph is used.
    # The KNNGraph is a graph representation, which uses the
    # KNNEdges edge definition with 8 neighbours as default.
    # The graph representation is defined by the detector,
    # in this case the Prometheus detector.
    # The standard node definition is used, which is NodesAsPulses.
    graph_definition = KNNGraph(detector=PONE())

    # Use GraphNetDataModule to load in data and create dataloaders
    # The input here depends on the dataset being used,
    # in this case the Prometheus dataset.
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

    training_dataloader = dm.train_dataloader
    validation_dataloader = dm.val_dataloader
    test_dataloader=dm.test_dataloader
    # Building model

    # Define architecture of the backbone, in this example
    # the DynEdge architecture is used.
    # https://iopscience.iop.org/article/10.1088/1748-0221/17/11/P11003
    backbone = DynEdge(
        nb_inputs=graph_definition.nb_outputs,
        global_pooling_schemes=["min", "max", "mean", "sum"],
    )
    # Define the task.
    # Here an energy reconstruction, with a LogCoshLoss function.
    # The target and prediction are transformed using the log10 function.
    # When infering the prediction is transformed back to the
    # original scale using 10^x.
    task = BinaryClassificationTask(
        hidden_size=backbone.nb_outputs,
        target_labels=config["target"],
        loss_function=BinaryCrossEntropyLoss(),
        transform_prediction_and_target=lambda x: x,
        transform_inference=lambda x: x,
    )
    # Define the full model, which includes the backbone, task(s),
    # along with typical machine learning options such as
    # learning rate optimizers and schedulers.
    # model = StandardModel(
    #     graph_definition=graph_definition,
    #     backbone=backbone,
    #     tasks=[task],
    #     optimizer_class=Adam,
    #     optimizer_kwargs={"lr": 1e-03, "eps": 1e-08},
    #     scheduler_class=PiecewiseLinearLR,
    #     scheduler_kwargs={
    #         "milestones": [
    #             0,
    #             len(training_dataloader) / 2,
    #             len(training_dataloader) * config["fit"]["max_epochs"],
    #         ],
    #         "factors": [1e-2, 1, 1e-02],
    #     },
    #     scheduler_config={
    #         "interval": "step",
    #     },
    # )
    # del model
    model_config = ModelConfig.load("/raven/ptmp/arego/Class/Training/R1T4K_100_parquet/Training_type_Mar-10_12-44/model_config.yml")
    model = Model.from_config(model_config,trust=True)  # With randomly initialised weights.
    model.load_state_dict("/raven/ptmp/arego/Class/Training/R1T4K_100_parquet/Training_type_Mar-10_12-44/state_dict.pth") 

    # # logger.info(f'check if models are identical :{modelA==model}')



    #Training model
    model.fit(
        training_dataloader,
        validation_dataloader,
        early_stopping_patience=config["early_stopping_patience"],
        logger=wandb_logger if wandb else None,
        **config["fit"],
    )

    # Get predictions
    additional_attributes =['event_no','record_id'] + model.target_labels 
    assert isinstance(additional_attributes, list)  # mypy

    logger.info(f'additional_attributes: {additional_attributes}')

    model.eval()

    # results = model.predict_as_dataframe(
    #     test_dataloader,
    #     prediction_columns=['out'],
    #     additional_attributes=additional_attributes,
    #     gpus=config["fit"]["gpus"],
    # )



    


    # Save predictions and model to file




    # Save results as .csv
    #results.to_csv(f"{path}/results.csv")

    # logger.info(f"shape: {results.shape}")
    # logger.info(f'head: {results.head().to_string()}')

    # Save full model (including weights) to .pth file - not version safe
    # Note: Models saved as .pth files in one version of graphnet
    #       may not be compatible with a different version of graphnet.
    model.save(f"/raven/ptmp/arego/Class/Training/R1T4K_100_parquet/Training_type_Mar-10_12-44/model.pth")

    # Save model config and state dict - Version safe save method.
    # This method of saving models is the safest way.
    model.save_state_dict(f"/raven/ptmp/arego/Class/Training/R1T4K_100_parquet/Training_type_Mar-10_12-44/state_dict.pth")
    model.save_config("/raven/ptmp/arego/Class/Training/R1T4K_100_parquet/Training_type_Mar-10_12-44/model_config_train_ds.yml/model_config.yml")
    

if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Train GNN model without the use of config files.
"""
    )

    parser.add_argument(
        "--path",
        help="Path to dataset file (default: %(default)s)",
        default=f"/raven/ptmp/arego/R1T4K_100_parquet",
    )

    parser.add_argument(
        "--pulsemap",
        help="Name of pulsemap to use (default: %(default)s)",
        default="hits",
    )

    parser.add_argument(
        "--target",
        default="duration",
    )

    parser.add_argument(
        "--truth-table",
        help="Name of truth table to be used (default: %(default)s)",
        default="records",
    )

    parser.with_standard_arguments(
        "gpus",
        ("max-epochs", 10),
        "early-stopping-patience",
        ("batch-size", 128),
        "num-workers",
    )

    parser.add_argument(
        "--wandb",
        action="store_true",
        help="If True, Weights & Biases are used to track the experiment.",
    )

    args, unknown = parser.parse_known_args()
    print(args)


    main(
        args.path,
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
