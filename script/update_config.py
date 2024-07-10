import argparse
import json

def update_config(config_path, parameter, value, num_value):
    with open(config_path, 'r') as file:
        config = json.load(file)

    if num_value is not None:
        config[parameter] = num_value
    else:
        config[parameter] = value

    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update configuration file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--parameter', type=str, required=True, help='Parameter to update')
    parser.add_argument('--value', type=str, required=False, help='New value for the parameter')
    parser.add_argument('--num_value', type=float, required=False, help='New value for the parameter')

    args = parser.parse_args()

    update_config(args.config, args.parameter, args.value, args.num_value)