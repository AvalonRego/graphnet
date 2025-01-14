# Example of reading events from Dataset class.

"""
Avalons comments:
what i think this script shows it how to load and intialize a model
"""
# Import necessary libraries
from timer import timer  # Timer utility to measure execution time
import awkward  # Awkward array for handling complex data structures (like jagged arrays)
import sqlite3  # SQLite database library
import time  # Standard time functions for sleep and timing
import torch.multiprocessing  # Multiprocessing utilities for torch
import torch.utils.data  # Data utilities from PyTorch
from torch_geometric.data.batch import Batch  # Graph batch processing utility from PyTorch Geometric
from tqdm import tqdm  # Progress bar utility

# GraphNet constants (file paths for the datasets)
#from graphnet.constants import TEST_PARQUET_DATA, TEST_SQLITE_DATA
# Feature and truth constants
from graphnet.data.constants import FEATURES, TRUTH
# Dataset class imports
from graphnet.data.dataset import Dataset
from graphnet.data.dataset import SQLiteDataset
from graphnet.data.dataset import ParquetDataset
# Argument parser and logger
from graphnet.utilities.argparse import ArgumentParser
from graphnet.utilities.logging import Logger
# Graph construction model
from graphnet.models.graphs import KNNGraph
# IceCube-specific detector class
from graphnet.models.detector.icecube import IceCubeDeepCore

# Dictionary mapping backend type to respective Dataset class
DATASET_CLASS = {
    "sqlite": SQLiteDataset,  # SQLite dataset class
    "parquet": ParquetDataset,  # Parquet dataset class
}

# Constants for feature and truth tables
features = FEATURES.PONE  # Feature set for DeepCore
truth = TRUTH.PONE  # Truth set for DeepCore

def main(backend: str) -> None:
    """Read intermediate file using `Dataset` class."""
    
    # Initialize logger to track events
    logger = Logger()

    # Check that the provided backend is valid
    assert backend in DATASET_CLASS

    # Set path for dataset based on the backend
    path = '/u/arego/project/Experimenting/data/graphnet_out/small/HexRealTracks.db'
    pulsemap = "hits"  # Pulse map name for detector data
    truth_table = "truth"  # Truth table name for truth values
    batch_size = 128  # Number of samples per batch
    num_workers = 30  # Number of worker processes for data loading
    wait_time = 0.00  # Time to wait between batches (in seconds)

    # Define the graph representation (using K-Nearest Neighbors graph)
    graph_definition = KNNGraph(detector=IceCubeDeepCore())

    # Loop over the tables (pulse map and truth)
    for table in ['hits']:
        # Get column names based on backend type
        if backend == "sqlite":
            with sqlite3.connect(path) as conn:
                cursor = conn.execute(f"SELECT * FROM {table} LIMIT 1")  # Fetch one record
                names = list(map(lambda x: x[0], cursor.description))  # Extract column names
        else:
            ak = awkward.from_parquet(path, lazy=True)  # Read Parquet file using Awkward Arrays
            names = ak[table].fields  # Get the fields (column names) from the specified table
            del ak  # Clean up the loaded data

        # Log the available columns in the table
        logger.info(f"Available columns in {table}")
        for name in names:
            logger.info(f"  . {name}")

    # Initialize the dataset using the correct backend class
    dataset = DATASET_CLASS[backend](
        path=path,
        pulsemaps=pulsemap,
        features=features,
        truth=truth,
        truth_table=truth_table,
        graph_definition=graph_definition,
    )
    # Ensure that the dataset is correctly instantiated
    assert isinstance(dataset, Dataset)

    # Log the dataset's first event and features
    logger.info(str(dataset[1]))
    logger.info(dataset[1].x)  # Log the features (x) of the first event
    if backend == "sqlite":
        assert isinstance(dataset, SQLiteDataset)  # Ensure it's a SQLite dataset if using SQLite
        # Close the SQLite connection if it was indexed
        dataset._close_connection()

    # Initialize the DataLoader for batching the data
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,  # Number of samples in each batch
        shuffle=True,  # Shuffle the data before batching
        num_workers=num_workers,  # Number of worker processes to load data in parallel
        collate_fn=Batch.from_data_list,  # Function to collate individual samples into a batch
        # persistent_workers=True,  # Uncomment for persistent workers in data loading (optional)
        prefetch_factor=2,  # Number of batches to prefetch in parallel
    )

    # Time the data loading process
    with timer("torch dataloader"):
        # Iterate through the DataLoader in batches
        for batch in tqdm(dataloader, unit=" batches", colour="green"):
            time.sleep(wait_time)  # Optional sleep between batches for performance tuning

    # Log information about the last batch
    logger.info(str(batch))  # Log the batch content
    logger.info(batch.size())  # Log the size of the batch
    logger.info(batch.num_graphs)  # Log the number of graphs in the batch

if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
Read a few events from data in an intermediate format.
"""
    )

    # Add an argument for selecting the backend (sqlite or parquet)
    parser.add_argument(
        "backend",
        choices=["sqlite", "parquet"],  # Valid backends to choose from
        default="sqlite",  # Default backend is sqlite
        const="sqlite",
        nargs="?",  # Argument is optional 
    )

    # Parse the arguments
    args, unknown = parser.parse_known_args()

    # Call the main function with the selected backend
    main(args.backend)
