import os
from typing import List, Optional
from datetime import datetime
import torch
from graphnet.utilities.logging import Logger
from graphnet.data import GraphNeTDataModule
from graphnet.data.dataset import SQLiteDataset, ParquetDataset
from graphnet.models import StandardModel, Model
from graphnet.utilities.config import ModelConfig
from graphnet.models.detector.pone import PONE  # PONE detector
from graphnet.models.gnn import RNN_TITO  # RNN-TITO backbone
from graphnet.models.graphs import KNNGraph
from graphnet.data.constants import FEATURES, TRUTH
from graphnet.models.graphs.nodes import NodeAsDOMTimeSeries 
def get_timestamp():
    return datetime.now().strftime("%b-%d_%H-%M")

def main(
    path: str,
    pulsemap: str,
    target: str,
    truth_table: str,
    gpus: Optional[List[int]],
    batch_size: int,
    num_workers: int,
    model_path: str,
) -> None:
    logger = Logger()
    archive = os.path.join('/raven/ptmp/arego/Class/', "Predictions")
    run_name = f"Predictions_{target}_{get_timestamp()}"
    
    features = FEATURES.PONE  # PONE features
    truth = TRUTH.PONE  # PONE truth

    dir_name=os.path.basename(path)
    run_name = f"Predictions_{target}_{dir_name}_{get_timestamp()}"

    config = {
        "path": path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "dataset_reference": SQLiteDataset if path.endswith(".db") else ParquetDataset,
    }
    graph_definition = KNNGraph(
        detector=PONE(),
        node_definition=NodeAsDOMTimeSeries(
            keys=features,
            id_columns=features[0:3],  # Spatial ID columns
            time_column=features[-1],  # Time column
            charge_column="None",
        ),
    )
    
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
    test_dataloader = dm.test_dataloader
    
    logger.info(f"Loading model from {model_path}")

    model_config = ModelConfig.load(f"{model_path}/model_config.yml")
    model = Model.from_config(model_config, trust=True)
    model.load_state_dict(torch.load(f"{model_path}/state_dict.pth"))

    
    additional_attributes = [target, "event_no", "record_id",'energy']
    results = model.predict_as_dataframe(
        test_dataloader,
        prediction_columns=['out'],
        additional_attributes=additional_attributes,
        gpus=gpus,
    )
    results["event_no"] = results["event_no"].astype("int64")
    
    path = os.path.join(archive, os.path.basename(model_path).split(".")[0], run_name)
    os.makedirs(path, exist_ok=True)
    results.to_csv(f"{path}/predictions.csv")
    
    logger.info(f"Predictions saved to {path}/predictions.csv")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load a pretrained model and generate predictions.")
    parser.add_argument("--path", required=True, help="Path to dataset file.")
    parser.add_argument("--pulsemap", default="hits", help="Name of pulsemap to use.")
    parser.add_argument("--target", required=True, help="Target variable.")
    parser.add_argument("--truth-table", default="records", help="Name of truth table.")
    parser.add_argument("--batch-size", type=int, default=68, help="Batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers.")
    parser.add_argument("--gpus", nargs="*", type=int, help="GPUs to use.")
    parser.add_argument("--model-path", required=True, help="Path to the pretrained model.")
    
    args = parser.parse_args()
    
    main(
        args.path,
        args.pulsemap,
        args.target,
        args.truth_table,
        args.gpus,
        args.batch_size,
        args.num_workers,
        args.model_path,
    )
