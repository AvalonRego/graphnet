from pone_extractor import PONE_H5HitExtractor

# Initialize the extractor
extractor = PONE_H5HitExtractor()

# Path to a sample .h5 file
test_file = '/u/arego/project/Experimenting/data/graphnet_test/small/HexRealTracks.h5'
# Run the extractor
data = extractor(test_file)

# Verify the output
if data is not None:
    print(data.head())
    print(data.keys())
else:
    print("No 'hits' table found in the file.")
