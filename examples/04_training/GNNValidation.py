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

torch.cuda.set_per_process_memory_fraction(0.5, device=0)
def get_timestamp():
    return datetime.now().strftime("%b-%d_%H-%M")

def main(dataset_path, pulsemap, target, truth_table, batch_size, num_workers, gpus, model_path, save_path):
    logger = Logger()
    run_name = f"Predictions_{target}_{os.path.basename(dataset_path)}_{get_timestamp()}"
    
    features = FEATURES.PONE
    truth = TRUTH.PONE
    
    config = {
        "path": dataset_path,
        "pulsemap": pulsemap,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "target": target,
        "dataset_reference": SQLiteDataset if dataset_path.endswith(".db") else ParquetDataset,
    }
    
    graph_definition = KNNGraph(detector=PONE())
    
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
    test_dataloader = dm.test_dataloader
    
    logger.info(f"Loading model from {model_path}")

    model_config = ModelConfig.load(f"{model_path}/model_config.yml")
    model = Model.from_config(model_config, trust=True)
    model.load_state_dict(torch.load(f"{model_path}/state_dict.pth"))
    
    additional_attributes = [target, "event_no", "record_id", "energy"]
    results = model.predict_as_dataframe(
        test_dataloader,
        prediction_columns=['out'],
        additional_attributes=additional_attributes,
        gpus=gpus,
    )
    results["event_no"] = results["event_no"].astype("int64")
    
    save_dir = os.path.join(save_path, os.path.basename(model_path).split(".")[0], run_name)
    os.makedirs(save_dir, exist_ok=True)
    results.to_csv(f"{save_dir}/predictions.csv")
    
    logger.info(f"Predictions saved to {save_dir}/predictions.csv")

if __name__ == "__main__":
    DATASET_PATH = "/u/arego/ptmp_link/TestTT"
    PULSEMAP = "hits"
    TARGET = "type"
    TRUTH_TABLE = "records"
    BATCH_SIZE = 32
    NUM_WORKERS = 12
    GPUS = [0]  # List of GPU IDs or None
    MODEL_PATH = "/u/arego/ptmp_link/Class/TrainingLoop/TwoData_2GPU_1/20250324_084749/"
    SAVE_PATH = "/raven/ptmp/arego/Class/TrackPredictions"
    
    main(DATASET_PATH, PULSEMAP, TARGET, TRUTH_TABLE, BATCH_SIZE, NUM_WORKERS, GPUS, MODEL_PATH, SAVE_PATH)
