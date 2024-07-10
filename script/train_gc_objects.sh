# Path to the configuration file
CONFIG_PATH="configs/default.json"

data_path="./data/gc_objects"

result_path="./results"
# Initialize an empty array
shape_ids=()

# Populate the array with the names of the subfolders
while IFS= read -r line; do
    shape_ids+=("$line")
done < <(find "$data_path" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

# Loop through shape IDs
for shape_id in "${shape_ids[@]}"; do
    # If the target folder exist, skip
    if [ -d "$result_path/${shape_id}_4prim_voxel" ]; then
        echo "Skipping $shape_id"
        continue
    fi
    # Update the configuration file
    python script/update_config.py --config "$CONFIG_PATH" --parameter shape_id --value "$shape_id"

    # Check if update_config.py executed successfully
    if [ $? -eq 0 ]; then
        # Run train.py
        python train.py --config_path "$CONFIG_PATH"

    else
        echo "Failed to update configuration for shape_id $shape_id"
        exit 1
    fi
done