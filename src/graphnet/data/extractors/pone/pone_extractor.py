"""H5 Extractor for PONE data files."""

from typing import List
import pandas as pd

from graphnet.data.extractors import Extractor


class PONE_H5Extractor(Extractor):
    """Class for extracting information from PONE h5 files."""

    def __init__(self, extractor_name: str, column_names: List[str]):
        """Construct H5Extractor.

        Args:
            extractor_name: Name of the `ParquetExtractor` instance.
            Used to keep track of the provenance of different data,
            and to name tables to which this data is saved.
            column_names: Name of the columns in `extractor_name`.
        """
        # Member variable(s)
        self._key = extractor_name
        self._column_names = column_names
        # Base class constructor
        super().__init__(extractor_name=extractor_name)

    def __call__(self, file_path: str) -> pd.DataFrame:
        """Extract information from h5 file."""

        with pd.HDFStore(file_path, mode='r') as hdf:
            # List all the keys in the file
            keys = hdf.keys()
            available_tables =[key for key in keys]
            #print(self._key,available_tables)
            if self._key in available_tables:
                #print('finally hit if')
                try:
                    #print('Inputs',file_path,self._key)
                    df=pd.read_hdf(file_path,key=self._key)
                except:
                    print('pandas has some issue?')
                self._verify_columns(df)
                df=df.rename(columns={'record_id':'event_no'}) #because thats how the other code calls it i guess
                return df 
            else:
                #print('hit else extractor')
                return None


    def _verify_columns(self, df: pd.core.frame.DataFrame) -> None:
        try:
            assert df.shape[1] == len(self._column_names)
        except AssertionError as e:
            self.error(
                f"Got {len(self._column_names)} column names but "
                f"{self._key} has {df.shape[1]}. Please make sure "
                f"that the column names match. ({self._column_names})"
            )
            raise e


class PONE_H5HitExtractor(PONE_H5Extractor):
    
#     """Extractor for `HitData` in PONE H5 files."""

    def __init__(self) -> None:
        """Extractor for `HitData` in PONE H5 files."""
        # Base class constructor
        super().__init__(
            extractor_name="/hits",
            column_names=['time', 
                          'string_id', 
                          'module_id', 
                          'pmt_id', 
                          'record_id', 
                          'type']
        )


class PONE_H5TruthExtractor(PONE_H5Extractor):
    """Extractor for `TruthData` in LiquidO H5 files."""

    def __init__(self) -> None:
        """Extractor for `TruthData` in LiquidO H5 files."""
        # Base class constructor
        super().__init__(
            extractor_name="/hits",
            column_names=['time', 
                          'string_id', 
                          'module_id', 
                          'pmt_id', 
                          'record_id', 
                          'type']
        )
        