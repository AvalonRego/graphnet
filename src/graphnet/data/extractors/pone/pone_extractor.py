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
        #print(self._key)
        self._column_names = column_names
        # Base class constructor
        super().__init__(extractor_name=extractor_name)


    def __call__(self, file_path: str) -> pd.DataFrame:
        """Extract information from h5 file."""


        df=None

        with pd.HDFStore(file_path, mode='r') as hdf:
            # List all the keys in the file
            keys = hdf.keys()
            available_tables =[key for key in keys]
            #print(self._key,available_tables)
            if self._key!='/hits':
                if self._key in available_tables:
                    #print('finally hit if')
                    try:
                        #print('Mod? Inputs',file_path,self._key)
                        df=pd.read_hdf(file_path,key=self._key)
                    except Exception as e:
                        print(f'Error occurred while reading HDF5 file: {e}')

                    #print(df.columns)
                    self._verify_columns(df)
                    #df=df.rename(columns={'record_id':'event_no'}) #because thats how the other code calls it i guess
                    #return df 
                else:
                    #print('hit else extractor')
                    df= None
            elif self._key=='/hits':
                self._key=["/hits",'/detector']

                df={}
                for key in self._key:
                    if key in available_tables:
                        #print('finally hit if')
                        try:
                            #print('Inputs',file_path,key)
                            df[key]=pd.read_hdf(file_path,key=key)
                            #df[key]=df[key].rename(columns={'record_id':'event_no'}) 
                        except Exception as e:
                            print(f'Error occurred while reading HDF5 file: {e}')

                df[self._key[0]]=df[self._key[0]][['time', 'string_id', 'module_id', 'pmt_id', 'event_no','record_id']]
                #print(df.keys())
                cols=['time', 'string_id', 'module_id', 
                          'pmt_id','record_id', 'event_no', 
                          'module_location_x', 
                          'module_location_y', 
                          'module_location_z', 
                          'pmt_orientation_x', 
                          'pmt_orientation_y', 
                          'pmt_orientation_z', 
                          'pmt_location_x', 
                          'pmt_location_y', 
                          'pmt_location_z']
                cols=[c for c in df[self._key[1]].columns if c in cols]
                df[self._key[1]]=df[self._key[1]][cols]
                df=pd.merge(df[self._key[0]],
                            df[self._key[1]],
                            on=['string_id', 'module_id', 'pmt_id'],
                            how='inner')
                self._verify_columns(df)
                self._key='/hits'
                #return df
                        
            else:
                print('something has gone terribly wrong')    
        return df                       

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
            extractor_name='/hits',
            column_names=['time', 'string_id', 'module_id', 
                          'pmt_id', 'event_no',
                          'record_id', 
                          'module_location_x', 
                          'module_location_y', 
                          'module_location_z', 
                          'pmt_orientation_x', 
                          'pmt_orientation_y', 
                          'pmt_orientation_z', 
                          'pmt_location_x', 
                          'pmt_location_y', 
                          'pmt_location_z']
        )


class PONE_H5TruthExtractor(PONE_H5Extractor):
    """Extractor for `TruthData` in LiquidO H5 files."""

    def __init__(self) -> None:
        """Extractor for `TruthData` in LiquidO H5 files."""
        # Base class constructor
        super().__init__(
            extractor_name="/records",
            column_names=['location_x', 'location_y', 'location_z',
                            'orientation_x','orientation_y', 'orientation_z',
                            'event_no','record_id', 'energy', 'length','time',
                            'type', 'particle_id', 'duration']
        )
        