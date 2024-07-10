# Base path for configuration files
CONFIG_BASE_PATH="configs"

# List of configuration files
config_files=("4_prim.json" "8_prim.json" "12_prim.json" "16_prim.json")
data_path="./data/gc_objects"
result_path="./results"

python script/update_config.py --config "$CONFIG_PATH" --parameter dataset_root --value "$data_path"
# Initialize an empty array
shape_ids=()

# Populate the array with the names of the subfolders
while IFS= read -r line; do
    shape_ids+=("$line")
done < <(find "$data_path" -mindepth 1 -maxdepth 1 -type d -exec basename {} \;)

# Loop through shape IDs
for shape_id in "${shape_ids[@]}"; do
    for config_file in "${config_files[@]}"; do
        CONFIG_PATH="$CONFIG_BASE_PATH/$config_file"
        
        config_number=$(echo "$config_file" | grep -oP '^\d+')

        # Construct the result folder name
        result_folder="${shape_id}_${config_number}_prim"
        
        # If the target folder exists, skip
        if [ -d "$result_path/$result_folder" ]; then
            echo "Skipping $shape_id with $config_file"
            continue
        fi
        
        # Update the configuration file
        python script/update_config.py --config "$CONFIG_PATH" --parameter shape_id --value "$shape_id"
        
        # Check if update_config.py executed successfully
        if [ $? -eq 0 ]; then
            # Run train.py
            python train.py --config_path "$CONFIG_PATH"
        else
            echo "Failed to update configuration for shape_id $shape_id with $config_file"
            exit 1
        fi
    done
done
            exit 1
        fi
    done
done
