# # Rename hits files without ~1~
# cd /raven/ptmp/arego/TestT/hits/
# for file in hits_*.parquet; do
#     [[ $file == *~1~ ]] && continue  # Skip files that have the ~1~ suffix
#     mv "$file" "temp_$file"
#     echo "Renamed: $file -> temp_$file"
# done

# hits_counter=0
# for file in $(ls temp_1_hits_*.parquet | sort -V); do
#     mv "$file" "hits_${hits_counter}.parquet"
#     echo "Renamed: $file -> hits_${hits_counter}.parquet"
#     ((hits_counter++))
# done


# echo "Hits renaming completed safely."

# # Rename hits files without ~1~
cd /raven/ptmp/arego/TestC/hits/
# for file in hits_*.parquet; do
#     [[ $file == *~1~ ]] && continue  # Skip files that have the ~1~ suffix
#     mv "$file" "temp_$file"
#     echo "Renamed: $file -> temp_$file"
# done

hits_counter=0
for file in $(ls temp_1_hits_*.parquet | sort -V); do
    mv "$file" "hits_${hits_counter}.parquet"
    echo "Renamed: $file -> hits_${hits_counter}.parquet"
    ((hits_counter++))
done


echo "Hits renaming completed safely."


