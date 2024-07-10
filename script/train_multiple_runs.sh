# Path to the configuration file
CONFIG_PATH="configs/default.json"

data_path="./data/gc_objects"

result_path="./results"
# Initialize an empty array
shape_ids=("ant")

# Loop through numbers 1 to 10
for i in {1..10}; do
    # Define experiment name
    experiment_name="run_$i"
    python script/update_config.py --config "$CONFIG_PATH" --parameter experiment_identifier --value "$experiment_name"
    for shape_id in "${shape_ids[@]}"; do
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
done
