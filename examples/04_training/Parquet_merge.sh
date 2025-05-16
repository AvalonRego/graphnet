#!/bin/bash

# Define the source directories
DIR1="/u/arego/ptmp_link/RC4K1_10_parquet"
DIR2="/u/arego/ptmp_link/R1T4K_EN10_parquet"

# Define the destination directory
DEST_DIR="/u/arego/ptmp_link/MIX10"
mkdir -p "$DEST_DIR/records" "$DEST_DIR/hits"

# Initialize counters for even and odd numbers
even_counter=2
odd_counter=1

# Process files in the first directory (even numbers)
for record_file in "$DIR1/records"/records_*.parquet; do
    # Extract the original number from the filename
    original_number=$(echo "$(basename "$record_file")" | grep -oP '\d+')

    # Construct the new even filename for records
    new_record_name="records_$even_counter.parquet"
    cp "$record_file" "$DEST_DIR/records/$new_record_name"

    # Construct the corresponding new even filename for hits
    new_hit_name="hits_$even_counter.parquet"
    cp "$DIR1/hits/hits_$original_number.parquet" "$DEST_DIR/hits/$new_hit_name"

    even_counter=$((even_counter + 2))
done

# Process files in the second directory (odd numbers)
for record_file in "$DIR2/records"/records_*.parquet; do
    # Extract the original number from the filename
    original_number=$(echo "$(basename "$record_file")" | grep -oP '\d+')

    # Construct the new odd filename for records
    new_record_name="records_$odd_counter.parquet"
    cp "$record_file" "$DEST_DIR/records/$new_record_name"

    # Construct the corresponding new odd filename for hits
    new_hit_name="hits_$odd_counter.parquet"
    cp "$DIR2/hits/hits_$original_number.parquet" "$DEST_DIR/hits/$new_hit_name"

    odd_counter=$((odd_counter + 2))
done

echo "Files have been copied and renamed to $DEST_DIR."
