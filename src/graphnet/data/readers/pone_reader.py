from typing import List, Union, Dict
from glob import glob
import os
import pandas as pd

from graphnet.data.extractors.pone import PONE_H5Extractor
from .graphnet_file_reader import GraphNeTFileReader

#print('PONE Reader')

class PONEReader(GraphNeTFileReader):
    """A class for reading h5 files from PONE."""

    _accepted_file_extensions = [".h5"]
    _accepted_extractors = [PONE_H5Extractor]

    def __call__(self, file_path: str) -> Dict[str, pd.DataFrame]:
        """Extract data from single h5 file.

        Args:
            file_path: Path to h5 file.

        Returns:
            Extracted data as a dictionary of DataFrames.
        """
        outputs = {}
        for extractor in self._extractors:
            try:
                #print(f'ponereader try extractor:{extractor}')
                output = extractor(file_path)
                #print(f'pone reader {output}')
                if output is not None:
                    outputs[extractor._extractor_name] = output
            except Exception as e:
                print(f"Error in extractor {extractor._extractor_name}: {e}")
        return outputs

    def find_files(self, path: Union[str, List[str]]) -> List[str]:
        """Search folder(s) for h5 files.

        Args:
            path: Directory or list of directories to search for h5 files.

        Returns:
            List of h5 files in the folders.
        """
        files = []
        if isinstance(path, str):
            path = [path]
        for p in path:
            files.extend(glob(os.path.join(p, "*.h5")))
        if not files:
            raise FileNotFoundError(f"No .h5 files found in the given path(s): {path}")
        return files
