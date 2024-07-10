import numpy as np
import utils
import os
import argparse

def inference_parameters(primitive_parameters_path, num_control_points=3, save_path="./inference_result"):
    """Produce sweep surfaces from primitive parameters."""
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    primitive_parameters = np.loadtxt(primitive_parameters_path).astype(np.float32)
    sweep_surfaces = utils.create_gt_swept_volume(
        primitive_parameters,
        num_control_points=num_control_points,
        file_prefix=save_path,
    )

def main():
    parser = argparse.ArgumentParser(description="Inference parameters for sweep surfaces.")
    parser.add_argument(
        "--primitive_parameters_path",
        type=str,
        default="./asset/primitive_parameters/octupus.txt",
        help="Path to the primitive parameters file.",
    )
    parser.add_argument(
        "--num_control_points",
        type=int,
        default=3,
        help="Number of control points for sweep surface generation.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./inference_result/octupus",
        help="Path to save the inference results.",
    )

    args = parser.parse_args()

    inference_parameters(
        primitive_parameters_path=args.primitive_parameters_path,
        num_control_points=args.num_control_points,
        save_path=args.save_path,
    )

if __name__ == "__main__":
    main()
