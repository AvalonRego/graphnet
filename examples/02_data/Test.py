
try:
    import torch
    print("Imported torch")
except ImportError as e:
    print(f"Failed to import torch:  ")

try:
    import torch.multiprocessing
    print("Imported torch.multiprocessing")
except ImportError as e:
    print(f"Failed to import torch.multiprocessing:  ")

try:
    import torch.utils.data
    print("Imported torch.utils.data")
except ImportError as e:
    print(f"Failed to import torch.utils.data:  ")

try:
    from torch_geometric.data.batch import Batch
    print("Imported Batch from torch_geometric.data.batch")
except ImportError as e:
    print(f"Failed to import Batch from torch_geometric.data.batch:  ")

try:
    from tqdm import tqdm
    print("Imported tqdm")
except ImportError as e:
    print(f"Failed to import tqdm:  ")

try:
    from graphnet.constants import TEST_PARQUET_DATA, TEST_SQLITE_DATA
    print("Imported constants from graphnet")
except ImportError as e:
    print(f"Failed to import constants from graphnet:  ")

try:
    from graphnet.data.constants import FEATURES, TRUTH
    print("Imported constants from graphnet.data")
except ImportError as e:
    print(f"Failed to import constants from graphnet.data:  ")

try:
    from graphnet.data.dataset import Dataset
    print("Imported Dataset from graphnet.data.dataset")
except ImportError as e:
    print(f"Failed to import Dataset from graphnet.data.dataset:  ")

try:
    from graphnet.data.dataset import SQLiteDataset
    print("Imported SQLiteDataset from graphnet.data.dataset")
except ImportError as e:
    print(f"Failed to import SQLiteDataset from graphnet.data.dataset:  ")

try:
    from graphnet.data.dataset import ParquetDataset
    print("Imported ParquetDataset from graphnet.data.dataset")
except ImportError as e:
    print(f"Failed to import ParquetDataset from graphnet.data.dataset:  ")

try:
    from graphnet.utilities.argparse import ArgumentParser
    print("Imported ArgumentParser from graphnet.utilities.argparse")
except ImportError as e:
    print(f"Failed to import ArgumentParser from graphnet.utilities.argparse:  ")

try:
    from graphnet.utilities.logging import Logger
    print("Imported Logger from graphnet.utilities.logging")
except ImportError as e:
    print(f"Failed to import Logger from graphnet.utilities.logging:  ")

try:
    from graphnet.models.graphs import KNNGraph
    print("Imported KNNGraph from graphnet.models.graphs")
except ImportError as e:
    print(f"Failed to import KNNGraph from graphnet.models.graphs:  ")

try:
    from graphnet.models.detector.icecube import IceCubeDeepCore
    print("Imported IceCubeDeepCore from graphnet.models.detector.icecube")
except ImportError as e:
    print(f"Failed to import IceCubeDeepCore from graphnet.models.detector.icecube:  ")


try:
    from timer import timer
    print("Imported timer")
except ImportError as e:
    print(f"Failed to import timer:  ")

try:
    import awkward
    print("Imported awkward")
except ImportError as e:
    print(f"Failed to import awkward:  ")

try:
    import sqlite3
    print("Imported sqlite3")
except ImportError as e:
    print(f"Failed to import sqlite3:  ")

try:
    import time
    print("Imported time")
except ImportError as e:
    print(f"Failed to import time:  ")
