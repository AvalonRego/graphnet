"""IceCube-specific `Detector` class(es)."""

from typing import Dict, Callable
import torch
import os

from graphnet.models.detector.detector import Detector
from graphnet.constants import PONE_GEOMETRY_TABLE_DIR

class PONE(Detector):
    """`Detector` class for PONE."""

    geometry_table_path = os.path.join(
        PONE_GEOMETRY_TABLE_DIR, "pone.parquet"
    )
    xyz = ['module_location_x', 'module_location_y','pmt_location_z']
    string_id_column = "string_id"
    module_id_column = "module_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            "module_location_x": self._dom_xyz,
            "module_location_y": self._dom_xyz,
            "module_location_z": self._dom_xyz,
            # 'module_radius', 
            # 'pmt_orientation_x',
            # 'pmt_orientation_y', 
            # 'pmt_orientation_z', 
            # 'pmt_id', 
            # 'pmt_efficiency',
            # 'pmt_area', 
            # 'pmt_noise_rate', 
        }
        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 500.0

    def _dom_time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4

    def _charge(self, x: torch.tensor) -> torch.tensor:
        return torch.log10(x)

    def _rde(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.25) / 0.25

    def _pmt_area(self, x: torch.tensor) -> torch.tensor:
        return x / 0.05