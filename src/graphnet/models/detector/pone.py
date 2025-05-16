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
    xyz = ['module_location_x', 'module_location_y','module_location_z']
    string_id_column = "string_id"
    module_id_column = "module_id"

    def feature_map(self) -> Dict[str, Callable]:
        """Map standardization functions to each dimension of input data."""
        feature_map = {
            'module_location_x':self._dom_xyz,
            'module_location_y':self._dom_xyz,
            'module_location_z':self._dom_xyz,
            'pmt_orientation_x':self._pmt_orientation,
            'pmt_orientation_y':self._pmt_orientation,
            'pmt_orientation_z':self._pmt_orientation,
            'pmt_location_x':self._pmt_location,
            'pmt_location_y':self._pmt_location,
            'pmt_location_z':self._pmt_location,
            'time':self._time,


        }
        return feature_map

    def _dom_xyz(self, x: torch.tensor) -> torch.tensor:
        return x / 1000.0

    def _pmt_orientation(self, x:torch.tensor)->torch.tensor:
        return (x+0.5963678105290182)/1.1927356210580364

    
    def _pmt_location(self, x: torch.tensor) -> torch.tensor:
        return x / 1000
    

    def _time(self, x: torch.tensor) -> torch.tensor:
        return (x - 1.0e04) / 3.0e4