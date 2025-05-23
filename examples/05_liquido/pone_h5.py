
import os
import gc

from graphnet.data.extractors.pone import PONE_H5HitExtractor,PONE_H5TruthExtractor
from graphnet.data.dataconverter import DataConverter
from graphnet.data.readers import PONEReader
from graphnet.data.writers import ParquetWriter, SQLiteWriter
from graphnet.utilities.argparse import ArgumentParser


def main(backend: str) -> None:
    """Convert h5 files from PONE to intermediate `backend` format."""
    # Fixed inputs
    input_dir = ["/raven/ptmp/arego/LBC1"]
    outdir = f"/raven/ptmp/arego/LBC1_parquet"
    os.makedirs(outdir, exist_ok=True)
    num_workers = 32

    if backend == "parquet":
        save_method = ParquetWriter()
    elif backend == "sqlite":
        save_method = SQLiteWriter()  # type: ignore

    converter = DataConverter(
        file_reader=PONEReader(),
        save_method=save_method,
        extractors=[PONE_H5HitExtractor(),PONE_H5TruthExtractor()],
        outdir=outdir,
        num_workers=num_workers,
    )

    converter(input_dir=input_dir)

    del converter
    gc.collect()

    #converter.merge_files()


if __name__ == "__main__":

    # Parse command-line arguments
    parser = ArgumentParser(
        description="""
            Convert h5 files from PONE to an intermediate format.
            """
    )

    parser.add_argument("backend", choices=["sqlite", "parquet"])

    args, unknown = parser.parse_known_args()

    # Run example script
    main(args.backend)