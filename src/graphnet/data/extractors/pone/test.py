from pone_extractor import H5HitExtractor

# Initialize the extractor
extractor = H5HitExtractor()

# Path to a sample .h5 file
test_file = '/u/arego/project/Experimenting/data/graphnet_test/HexRealTracks.h5'
# Run the extractor
data = extractor(test_file)

# Verify the output
if data is not None:
    print(data.head())
else:
    print("No 'hits' table found in the file.")
