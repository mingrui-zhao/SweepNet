"""Utility functions for training and testing."""

import os
import pathlib
import time
import mcubes
import numpy as np
import pyvista as pv
import torch
from gpytoolbox.copyleft import swept_volume
import shapely
import trimesh
from gpytoolbox import read_mesh, write_mesh
from gpytoolbox.copyleft import swept_volume
import open3d as o3d
from PIL import Image
import imageio

np.random.seed(42)
torch.manual_seed(42)


# pylint: disable=W0632
### Experiment utilities
def init(config):
    """Initialise the experiment directory and copy the code to the directory."""
    if not os.path.exists(f"./{config.sample_dir}/{config.experiment_name}"):
        pathlib.Path(f"./{config.sample_dir}/{config.experiment_name}").mkdir(
            parents=True, exist_ok=True
        )


### Bspline utilities
class BsplineCache:
    """Bspline cache class"""

    def __init__(self, cache_file="bspline_cache.pt"):
        self.cache_file = cache_file
        self.cache = self.load_cache()

    def load_cache(self):
        """Load the cache from the cache file."""
        if os.path.exists(self.cache_file):
            return torch.load(self.cache_file)
        return {}

    def save_cache(self):
        """Save the cache to the cache file."""
        torch.save(self.cache, self.cache_file)

    def get_key(self, n, k, p):
        """Get precomputed Bspline key."""
        return f"{n}-{k}-{p}"

    def spline_basis_matrix(self, knots, t, n=3, k=2):
        """Bspline basis matrix for n control points, degree k, preset knots and resolution t."""
        num_parameters = len(t)
        result_matrix = torch.zeros((num_parameters, n))
        for i in range(n):
            for j, ti in enumerate(t):
                result_matrix[j, i] = basis_function(i, k, ti, knots)
        return result_matrix

    def evaluate_bspline_coefficient(self, n, k, p):
        """Evalute the Bspline coefficient."""
        knots = (
            torch.concatenate(
                (torch.zeros(k), torch.linspace(0, 1, n - k + 1), torch.ones(k))
            )
            .to(torch.float32)
            .cuda()
        )
        t = (
            torch.cat((torch.linspace(0, 0.99, p - 1), torch.tensor([0.9999])))
            .to(torch.float32)
            .cuda()
        )
        return self.spline_basis_matrix(knots, t, n=n, k=k)

    def get_bspline_coefficient(self, n, k, p):
        """Query bspline coefficient from cache or compute and cache if not found."""
        key = self.get_key(n, k, p)
        if key in self.cache:
            # If the result is a tensor, ensure it's moved to the correct device before returning
            return self.cache[key].to("cuda" if torch.cuda.is_available() else "cpu")

        # If not in cache, compute, save, and return
        result = self.evaluate_bspline_coefficient(n, k, p)
        self.cache[key] = (
            result.cpu()
        )  # Convert to CPU tensor before caching to avoid CUDA memory leak
        self.save_cache()
        return result.to("cuda" if torch.cuda.is_available() else "cpu")


def basis_function(i, k, t, knots):
    """Recursive implementation of basis function for Bsplines with degree k."""
    if k == 0:
        if knots[i] <= t < knots[i + 1]:
            return 1
        return 0
    denominator1 = knots[i + k] - knots[i]
    if denominator1 == 0:
        term1 = 0
    else:
        term1 = ((t - knots[i]) / denominator1) * basis_function(i, k - 1, t, knots)
    denominator2 = knots[i + k + 1] - knots[i + 1]
    if denominator2 == 0:
        term2 = 0
    else:
        term2 = ((knots[i + k + 1] - t) / denominator2) * basis_function(
            i + 1, k - 1, t, knots
        )
    return term1 + term2


def spline_basis_matrix(knots, t, n=3, k=2):
    """Bspline basis matrix for n control points, degree k, preset knots and resolution t."""
    num_parameters = len(t)
    result_matrix = torch.zeros((num_parameters, n))
    for i in range(n):
        for j, ti in enumerate(t):
            result_matrix[j, i] = basis_function(i, k, ti, knots)
    return result_matrix


### Visualization utilities
def get_distinct_color(index):
    """
    Returns a distinct color from a predefined palette of 16 colors.
    Cycles through the palette if the index is out of bounds.
    """
    palette = [
        [0.33, 0.42, 0.18],  # Cactus Green
        [0.70, 0.00, 0.00],  # Crimson Red
        [0.53, 0.81, 0.92],  # Sky Blue
        [0.94, 0.77, 0.06],  # Sunflower Yellow
        [0.85, 0.44, 0.84],  # Orchid Purple
        [1.00, 0.50, 0.31],  # Coral Orange
        [0.00, 0.50, 0.50],  # Teal Blue
        [0.93, 0.23, 0.51],  # Magenta
        [0.77, 0.93, 0.00],  # Lime Green
        [0.00, 0.00, 0.50],  # Navy Blue
        [0.54, 0.21, 0.06],  # Burnt Sienna
        [0.44, 0.50, 0.56],  # Slate Grey
        [0.70, 0.50, 0.82],  # Lavender
        [1.00, 0.85, 0.70],  # Peach
        [0.50, 0.50, 0.00],  # Olive Green
        [0.21, 0.27, 0.31],  # Charcoal Grey
    ]
    # Normalize colors to the range [0, 1]
    normalized_palette = [[c * 255 for c in color] for color in palette]
    return np.array(normalized_palette[index % len(normalized_palette)]).astype(
        np.uint8
    )


def polyline_from_points(points):
    """Generate polygonal lines from points for Pyvista package."""
    # Ensure points is a numpy array and has the right shape
    points = np.array(points, dtype=np.float64)
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    
    # Handle empty or invalid points
    if len(points) == 0:
        return pv.PolyData()
    
    # Ensure points have the right shape (N, 3)
    if points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
    
    # Check for NaN or infinity values
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        print(f"Warning: Points contain NaN or infinity values, filtering them out")
        valid_mask = np.isfinite(points).all(axis=1)
        points = points[valid_mask]
        
        if len(points) == 0:
            print("Warning: No valid points after filtering NaN/infinity values")
            return pv.PolyData()
    
    # Create PolyData using the proper method
    try:
        # Use pyvista's spline creation which handles all internal attributes properly
        if len(points) > 1:
            poly = pv.Spline(points)
        else:
            # Single point case
            poly = pv.PolyData(points)
    except Exception as e:
        print(f"Error creating spline: {e}")
        # Fallback: try the older method with proper initialization
        try:
            poly = pv.PolyData()
            poly.points = points
            if len(points) > 1:
                the_cell = np.arange(0, len(points), dtype=np.int_)
                the_cell = np.insert(the_cell, 0, len(points))
                poly.lines = the_cell
        except Exception as e2:
            print(f"Fallback method also failed: {e2}")
            # Final fallback: create with minimal lines structure
            try:
                if len(points) > 1:
                    lines = []
                    for i in range(len(points) - 1):
                        lines.extend([2, i, i + 1])
                    poly = pv.PolyData(points, lines=lines)
                else:
                    poly = pv.PolyData(points)
            except Exception as e3:
                print(f"Final fallback failed: {e3}")
                raise e
    
    return poly


def render_general_cyl(
    profile_axis, outpath, n=3, k=2, tube_save=True, color_option=None
):
    """Render generalized cylinders (sweeping axis) with different colors."""
    control_points = profile_axis.reshape(-1, n, 3)
    knots = torch.concatenate(
        (torch.zeros(k), torch.linspace(0, 1, n - k + 1), torch.ones(k))
    )
    # Define the parameter values
    t = torch.linspace(knots[k], knots[-k - 1], 100)  # type: ignore
    basis_mat = spline_basis_matrix(knots, t, n=n, k=k)
    
    # Convert tensors to numpy arrays for consistent operations
    basis_mat = basis_mat.detach().cpu().numpy()
    control_points = np.array(control_points)
    
    all_gc = pv.PolyData()
    for i in range(control_points.shape[0]):
        curve_points = np.matmul(basis_mat, control_points[i])[:-1]
        
        # Skip if we don't have enough points
        if len(curve_points) < 2:
            print(f"Warning: Skipping primitive {i} - insufficient points ({len(curve_points)})")
            continue
            
        polyline = polyline_from_points(curve_points)
        
        # Skip if polyline creation failed
        if polyline.n_points == 0:
            print(f"Warning: Skipping primitive {i} - failed to create polyline")
            continue
            
        polyline["scalars"] = np.arange(polyline.n_points)
        
        try:
            tube = polyline.tube(radius=0.005)
        except Exception as e:
            print(f"Warning: Failed to create tube for primitive {i}: {e}")
            continue
            
        if color_option is None:  # Distinct color
            color = get_distinct_color(i * 2)
        else:  # Coral color
            color = np.array([240.0, 128.0, 128.0])
        tube.point_data["RGB"] = np.full((tube.n_points, 3), np.uint8(color))  # type: ignore
        if tube_save:
            try:
                tube.save(
                    os.path.join(os.path.dirname(outpath), f"spline_{i}.ply"), texture="RGB"
                )
            except Exception as e:
                print(f"Warning: Failed to save tube for primitive {i}: {e}")
        all_gc += tube  # type: ignore
    print(f"Saving the rendered general cylinder representations to {outpath}")
    all_gc.save(outpath, texture="RGB")


### Evaluation utilities
def generate_testing_points(model_resolution):
    """
    Generate a set of testing points within a 3D space.

    Parameters:
    - model_resolution (int): The resolution of the model.

    Returns:
    - points (numpy.ndarray): An array of testing points with shape (N, 3),
      where N is the total number of points.
    """
    points = np.indices(
        (model_resolution, model_resolution, model_resolution)
    ).T.reshape(-1, 3)
    points = (points + 0.5) / model_resolution - 0.5
    return points


def generate_chunked_testing_points(model_resolution, eval_resolution):
    """
    Generate chunked testing points based on the model resolution and evaluation resolution.

    Parameters:
        model_resolution (int): The resolution of the model.
        eval_resolution (int): The resolution for evaluation.

    Returns:
        list: A list of chunked testing points.
    """
    testing_points = np.array_split(
        generate_testing_points(model_resolution),
        int(model_resolution / eval_resolution) ** 3,
        axis=0,
    )
    return testing_points


def create_gt_swept_volume(
    primitive_parameters, num_control_points=3, file_prefix=None, edit_suffix=None
):
    """Generate ground truth sweep surfaces from primitive parameters."""
    primitive_parameters = torch.from_numpy(primitive_parameters)
    import pdb; pdb.set_trace()
    num_primitives = primitive_parameters.shape[1]
    control_points = primitive_parameters[:, :, : 3 * num_control_points]
    
    # Convert control points to the format expected by render_general_cyl
    control_points_np = control_points.detach().cpu().numpy()
    
    if edit_suffix is not None:
        render_general_cyl(
            control_points_np,
            os.path.join(file_prefix, f"spline_axis_{edit_suffix}.ply"),
            n=num_control_points,
        )
    else:
        render_general_cyl(
            control_points_np,
            os.path.join(file_prefix, f"spline_axis.ply"),
            n=num_control_points,
        )
    scale_term = primitive_parameters[:, :, 3 * num_control_points + 3 :].reshape(-1, 2)
    transform = get_parallel_transport_frame(
        control_points, 25, scale_term, n=num_control_points, k=2
    )
    combined_mesh = None
    for i in range(num_primitives):
        _, contour, _ = sample_scaling_superellipse_points(
            primitive_parameters[:, i, :].unsqueeze(0),
            n=num_control_points,
            loop_points=100,
        )
        contour = np.array(contour.squeeze(0, 1).cpu().numpy())
        contour = np.concatenate((contour, [contour[0]]))
        polygon = shapely.Polygon(contour)
        if edit_suffix is not None:
            render_general_cyl(
                control_points_np[:, i, :][:, None, :],
                os.path.join(file_prefix, f"spline_axis_{i}_{edit_suffix}.ply"),
                n=num_control_points,
            )
        else:
            render_general_cyl(
                control_points_np[:, i, :][:, None, :],
                os.path.join(file_prefix, f"spline_axis_{i}.ply"),
                n=num_control_points,
            )
        mesh = trimesh.creation.extrude_polygon(polygon, height=1 / 64)
        if edit_suffix is not None:
            profile_path = os.path.join(file_prefix, f"profile_{i}_{edit_suffix}.ply")
        else:
            profile_path = os.path.join(file_prefix, f"profile_{i}.ply")
        mesh.export(profile_path, file_type="ply")
        matrices = transform.squeeze(0).detach().cpu().numpy()
        transmat = matrices[i]
        inverted_matrices = np.array([np.linalg.inv(matrix) for matrix in transmat])
        v, f = read_mesh(profile_path)
        u, g = swept_volume(
            v,
            f,
            transformations=inverted_matrices,
            eps=0.01,
            verbose=False,
            align_rotations_with_velocity=False,
        )
        if edit_suffix is not None:
            gt_path = os.path.join(file_prefix, f"gt_sweep_{i}_{edit_suffix}.obj")
        else:
            gt_path = os.path.join(file_prefix, f"gt_sweep_{i}.obj")
        write_mesh(gt_path, u, g)
        mesh = o3d.io.read_triangle_mesh(gt_path)
        # Assign a unique color
        mesh.paint_uniform_color(
            (np.array(get_distinct_color(i)).astype(np.float64)) / 255
        )
        # Combine meshes
        if combined_mesh is None:
            combined_mesh = mesh
        else:
            combined_mesh += mesh
    if edit_suffix is not None:
        combined_file_path = os.path.join(file_prefix, f"union_sweep_{edit_suffix}.obj")
    else:
        combined_file_path = os.path.join(file_prefix, "union_sweep.obj")
    o3d.io.write_triangle_mesh(combined_file_path, combined_mesh)


def generate_sv_video_from_param(
    primitive_parameters,
    num_control_points=3,
    file_prefix=None,
    edit_suffix=None,
    num_frames=60,
):
    """Visualise the sweeping process with primitive parameters as input."""
    primitive_parameters = torch.from_numpy(primitive_parameters).unsqueeze(0)
    num_primitives = primitive_parameters.shape[1]
    control_points = primitive_parameters[0:, 0:, : 3 * num_control_points]
    render_general_cyl(
        control_points,
        os.path.join(file_prefix, f"spline_axis_{edit_suffix}.ply"),
        n=num_control_points,
        color_option="coral",
    )

    scale_term = primitive_parameters[0:, 0:, 3 * num_control_points + 3 :].reshape(
        -1, 2
    )
    transform = get_parallel_transport_frame(
        control_points, 120, scale_term, n=num_control_points, k=2
    )

    output_dict = {}
    for i in range(num_primitives):
        _, contour, _ = sample_scaling_superellipse_points(
            primitive_parameters[:, i, :].unsqueeze(0),
            n=num_control_points,
            loop_points=100,
        )
        contour = np.array(contour.squeeze(0, 1).cpu().numpy())
        contour = np.concatenate((contour, [contour[0]]))
        polygon = shapely.Polygon(contour)
        mesh = trimesh.creation.extrude_polygon(polygon, height=1 / 64)
        profile_path = os.path.join(file_prefix, f"profile_{i}_{edit_suffix}.ply")
        mesh.export(profile_path, file_type="ply")
        matrices = transform.squeeze(0).detach().cpu().numpy()
        transmat = matrices[i]
        inverted_matrices = np.array([np.linalg.inv(matrix) for matrix in transmat])
        v, f = read_mesh(profile_path)
        output_dict[i] = {"v": v, "f": f, "inverted_matrices": inverted_matrices}
        os.remove(profile_path)

    for frame in range(num_frames):
        print(f"Generating frame {frame + 1}/{num_frames}")
        # combined_mesh = None
        combined_mesh = trimesh.load(f"{file_prefix}/spline_axis_{edit_suffix}.ply")
        progress = (frame + 1) / num_frames
        for i in range(num_primitives):
            v, f = output_dict[i]["v"], output_dict[i]["f"]
            inverted_matrices = output_dict[i]["inverted_matrices"]
            current_matrices = inverted_matrices[
                : int(progress * len(inverted_matrices))
            ]
            u, g = swept_volume(
                v,
                f,
                transformations=current_matrices,
                eps=0.01,
                verbose=False,
                align_rotations_with_velocity=False,
            )

            mesh = trimesh.Trimesh(vertices=u, faces=g)
            # Assign a unique color
            face_color = np.concatenate((get_distinct_color(i), [255]), -1)
            mesh.visual.face_colors = np.tile(face_color, (mesh.faces.shape[0], 1))

            # Combine meshes
            if combined_mesh is None:
                combined_mesh = mesh
            else:
                combined_mesh = trimesh.util.concatenate([combined_mesh, mesh])

        frame_file_path = os.path.join(
            file_prefix, f"frame_{frame:03d}_{edit_suffix}.obj"
        )
        combined_mesh.export(frame_file_path, file_type="obj")


def generate_mesh_primitives(model, voxel, config, bspline_cache=None, n=3, k=2):
    """Generate sweep surfaces primitives from the model."""
    feature = model.encoder(voxel)
    code = model.decoder(feature)
    union_layer_connections = model.selection_head(code, is_training=False)
    primitive_parameters = model.swept_volume_head(code)

    file_prefix = os.path.join(*[config.sample_dir, config.experiment_name])

    compound_mask = union_layer_connections * model.neural_sweeper.union_mask

    mask = (compound_mask == 1).flatten()
    primitive_parameters_selected = primitive_parameters[:, mask, :]
    np.savetxt(
        os.path.join(file_prefix, "primitive_parameters.txt"),
        primitive_parameters_selected[0].detach().cpu().numpy(),
    )
    control_points = primitive_parameters[:, mask, : 3 * n]
    scale_term = primitive_parameters_selected[:, :, 3 * n + 3 :].reshape(-1, 2)
    transform = get_parallel_transport_frame(
        control_points, 25, scale_term, n=n, k=k, bspline_cache=None
    )
    matrices = transform.detach().cpu().numpy()
    unioned_mesh = []
    combined_mesh = None

    render_general_cyl(
        control_points.detach().cpu().numpy(),
        os.path.join(file_prefix, "spline_axis.ply"),
        n=n,
    )

    for i, primitive_param in enumerate(primitive_parameters_selected[0]):
        _, contour, _ = sample_scaling_superellipse_points(
            primitive_param.unsqueeze(0),
            n=n,
            loop_points=100,
            bspline_cache=bspline_cache,
        )
        contour = np.array(contour.squeeze(0, 1).cpu().numpy())
        contour = np.concatenate((contour, [contour[0]]))
        polygon = shapely.Polygon(contour)
        # Extrude the path to create a 3D mesh
        mesh = trimesh.creation.extrude_polygon(polygon, height=1 / 64)
        mesh.export(os.path.join(file_prefix, f"profile_{i}.ply"), file_type="ply")
        v = mesh.vertices
        f = mesh.faces
        transmat = matrices[i]
        inverted_matrices = np.array([np.linalg.inv(matrix) for matrix in transmat])
        u, g = swept_volume(
            v,
            f,
            transformations=inverted_matrices,
            eps=0.01,
            verbose=False,
            align_rotations_with_velocity=False,
        )
        mesh = trimesh.Trimesh(vertices=u, faces=g)
        face_color = np.concatenate((get_distinct_color(i), [255]), -1)
        mesh.visual.face_colors = np.tile(face_color, (mesh.faces.shape[0], 1))
        mesh.export(os.path.join(file_prefix, f"gt_sweep_{i}.ply"))
        if combined_mesh is None:
            combined_mesh = mesh
        else:
            # For combining without boolean union, you can simply add the vertices and faces
            combined_mesh = trimesh.util.concatenate([combined_mesh, mesh])
        unioned_mesh.append(mesh)

    unioned_mesh = trimesh.boolean.boolean_manifold(
        unioned_mesh, "union", check_volume=False
    )

    combined_mesh_path = os.path.join(file_prefix, "gt_sweep.ply")
    unioned_mesh_path = os.path.join(file_prefix, "unioned_sweep.ply")
    # Perform Boolean union operation

    # Save the combined mesh
    combined_mesh.export(combined_mesh_path)
    print(f"Combined mesh saved to: {combined_mesh_path}")

    # Save the unioned mesh
    unioned_mesh.export(unioned_mesh_path)
    print(f"Unioned mesh saved to: {unioned_mesh_path}")


### Loss utilities
def calculate_scores_and_mask(primitive_occupancies):
    B, K, N = primitive_occupancies.shape
    scores = torch.zeros((B, K))
    mask = torch.zeros((B, K))

    for b in range(B):  # iterate over each sample in the batch
        for k in range(K):  # iterate over each primitive
            total_occupancy = torch.sum(primitive_occupancies[b, k] > 0.5)

            # Check if primitive is empty
            if total_occupancy < 0.001 * N:
                scores[b, k] = torch.inf
                continue

            unique_score = calculate_unique_points(primitive_occupancies[b], k)

            # Calculate score and update mask
            scores[b, k] = unique_score
            mask[b, k] = 1

    return scores.to(primitive_occupancies.device), mask.to(
        primitive_occupancies.device
    )


def calculate_unique_points(primitive_occupancies, k):

    primitive_occupancies_excluded = primitive_occupancies.clone()
    primitive_occupancies_excluded[k] = 0.0
    occupancies_excluded = torch.clamp(
        torch.sum(primitive_occupancies_excluded > 0.5, dim=0), 0.0, 1.0
    )

    occupancies_full = torch.clamp(
        torch.sum(primitive_occupancies > 0.5, dim=0), 0.0, 1.0
    )
    num_unique_points = torch.sum(occupancies_full - occupancies_excluded)
    unique_score = num_unique_points / torch.sum(primitive_occupancies[k] > 0.5)

    return unique_score


### Sweep surface utilities
def sample_scaling_superellipse_points(
    sweep_param,
    n=3,
    k=2,
    spline_points=124,
    loop_points=100,
    num_loops=10,
    bspline_cache=None,
):
    """Sample points on a superellipse."""
    device = sweep_param.device
    if len(sweep_param.shape) == 3:
        B, N, K = sweep_param.shape
        sweep_param = sweep_param.reshape(B * N, K)
        num_primitives = B * N
    else:
        assert len(sweep_param.shape) == 2
        N, K = sweep_param.shape
        num_primitives = N
    control_points = sweep_param[:, : n * 3].reshape(num_primitives, n, 3)

    if bspline_cache is None:
        knots = torch.concatenate(
            (torch.zeros(k), torch.linspace(0, 1, n - k + 1), torch.ones(k))
        ).to(control_points.device)
        t_values = torch.linspace(
            0.0, 0.9999, spline_points + 1
        )  # slightly less than 1 to ensure it's within the domain
        # Axis points

        bspline_basis = spline_basis_matrix(knots, t_values, n=n, k=k).to(control_points.device)
        bspline_points = torch.matmul(bspline_basis, control_points)
    else:
        bspline_coe = bspline_cache.get_bspline_coefficient(n, k, spline_points + 1).to(control_points.device)
        bspline_points = torch.matmul(bspline_coe, control_points)
    # Profile parameters
    a = sweep_param[:, n * 3 : n * 3 + 1]
    b = sweep_param[:, n * 3 + 1 : n * 3 + 2]

    deg = sweep_param[:, n * 3 + 2 : n * 3 + 3]

    if sweep_param.shape[1] > n * 3 + 3:
        scale_term = sweep_param[:, n * 3 + 3 :]
    else:
        scale_term = None

    t = torch.linspace(0, 2 * torch.pi, loop_points).to(device).unsqueeze(0)

    # Profile coordinates
    x = (
        a
        * torch.tanh(torch.cos(t) * 100)
        * torch.pow(torch.abs(torch.cos(t)), (2 / deg))
    )
    y = (
        b
        * torch.tanh(torch.sin(t) * 100)
        * torch.pow(torch.abs(torch.sin(t)), (2 / deg))
    )

    parallel_transforms = get_parallel_transport_frame(
        control_points,
        num_loops,
        scale_term=scale_term,
        n=n,
        k=k,
        bspline_cache=bspline_cache,
    )
    transform = torch.inverse(parallel_transforms)

    profile_points = torch.stack((x, y), dim=-1)
    profile_points_2d_aug = torch.zeros(num_primitives, loop_points, 1).to(device)
    profile_points_homo_aug = torch.ones(num_primitives, loop_points, 1).to(device)
    profile_points_homo = torch.cat(
        (profile_points, profile_points_2d_aug, profile_points_homo_aug), dim=-1
    )
    profile_points_transformed = torch.matmul(
        transform.unsqueeze(2), profile_points_homo[:, None, :, :, None]
    ).squeeze(-1)[
        ..., :3
    ]  # num_primitives x num_loops x loop points x 3
    profile_points_transformed = profile_points_transformed.view(num_primitives, -1, 3)

    return bspline_points, profile_points, profile_points_transformed


def vis_mesh(model, input3d, vis_res, iter_res, epoch, config):
    test_points = generate_chunked_testing_points(vis_res, iter_res)
    model_occ = []
    for index, item in enumerate(test_points):
        model_occ.append(
            model(
                input3d,
                torch.tensor(item, dtype=torch.float32).unsqueeze(0).to("cuda"),
                is_training=False,
            )[0]
        )
    model_occ = torch.cat(model_occ, dim=0)
    model_occ = model_occ.view(vis_res, vis_res, vis_res).permute(2, 1, 0)
    model_occ = model_occ.cpu().detach().numpy()
    model_occ = (model_occ > 0.5).astype(np.uint8)
    model_occ = mcubes.smooth(model_occ)
    vert, face = mcubes.marching_cubes(model_occ, 0)
    vert = (vert / vis_res) - 0.5
    mesh = trimesh.Trimesh(vertices=vert, faces=face)
    mesh.export(
        f"./{config.sample_dir}/{config.experiment_name}/mesh_epoch_{epoch}.ply"
    )


def get_parallel_transport_frame(
    control_points,
    num_samples,
    scale_term,
    n=3,
    k=2,
    bspline_cache=None,
):
    """Obtain parallel transport frame transformation matrix from a Bspline."""
    B, K, N = control_points.shape
    control_points = control_points.reshape(
        -1, n, 3
    )  # Num primitives x Num control points x 3

    knots = torch.concatenate(
        (torch.zeros(k), torch.linspace(0, 1, n - k + 1), torch.ones(k))
    ).to(control_points.device)

    t_values = torch.cat(
        (torch.linspace(0.0, 0.99, num_samples), torch.tensor([0.9999]))
    ).to(
        control_points.device
    )  # slightly less than 1 to ensure it's within the domain
    scale_trace = scale_term[:, 0:1] * t_values**2 + scale_term[:, 1:2] * t_values + 1.0

    if bspline_cache is None:
        bspline_basis = spline_basis_matrix(knots, t_values, n=n, k=k).to(
            control_points.device
        )
        bspline_points = torch.matmul(bspline_basis, control_points)
    else:
        bspline_coe = bspline_cache.get_bspline_coefficient(n, k, num_samples + 1)
        bspline_points = torch.matmul(bspline_coe, control_points)

    scale_trace = scale_trace[:, :-1]
    z_axis = bspline_points[:, 1, :] - bspline_points[:, 0, :]
    z_axis = z_axis / torch.norm(z_axis, dim=-1, keepdim=True)

    # Define an arbitrary orthogonal x-axis
    v = torch.zeros_like(z_axis)
    v[:, 0] = -1 * z_axis[:, 1]
    v[:, 1] = z_axis[:, 0]

    x_axis = torch.cross(z_axis, v, dim=-1)
    x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

    y_axis = torch.cross(x_axis, z_axis, dim=-1)
    y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True)
    frames = [torch.stack((y_axis, x_axis, z_axis), dim=-2)]

    for i in range(1, bspline_points.shape[-2] - 1):
        # Compute new z-axis
        z_axis = bspline_points[:, i + 1, :] - bspline_points[:, i, :]
        z_axis = z_axis / torch.norm(z_axis, dim=-1, keepdim=True)

        # Compute the new x-axis as the cross product between new z-axis and the previous y-axis
        x_axis = torch.cross(z_axis, y_axis, dim=-1)
        x_axis = x_axis / torch.norm(x_axis, dim=-1, keepdim=True)

        # Compute new y-axis as the cross product of new z and x axes
        y_axis = torch.cross(x_axis, z_axis, dim=-1)
        y_axis = y_axis / torch.norm(y_axis, dim=-1, keepdim=True)
        frames.append(torch.stack((y_axis, x_axis, z_axis), dim=-2))

    rotation_matrix = torch.stack(
        frames, dim=1
    )  # Shape: [Num primitives x Num nodes x 3 x 3]

    translation_matrix = -1 * torch.matmul(
        rotation_matrix,
        (bspline_points[:, :-1, :]).unsqueeze(-1),
    )
    rotation_matrix = rotation_matrix * torch.pow(scale_trace + 1e-4, -1).unsqueeze(
        -1
    ).unsqueeze(-1)
    translation_matrix = translation_matrix * torch.pow(
        scale_trace + 1e-4, -1
    ).unsqueeze(-1).unsqueeze(-1)
    transform = torch.cat((rotation_matrix, translation_matrix), dim=-1)

    # Create the last row for each transformation matrix
    last_rows = torch.zeros(transform.shape[0], transform.shape[1], 1, 4).to(
        transform.device
    )
    last_rows[:, :, :, 3] = 1

    # Concatenate the last row to each transformation matrix
    transform = torch.cat((transform, last_rows), dim=2)
    return transform


def save_model_modules(model, save_path, exclude_neural_sweeper=True):
    """
    Save only the trainable modules of the model, excluding the neural sweeper for efficiency.
    
    Args:
        model: The SweepNet model
        save_path: Path to save the checkpoint
        exclude_neural_sweeper: Whether to exclude neural sweeper from saving
    """
    checkpoint = {
        "encoder_state_dict": model.encoder.state_dict(),
        "decoder_state_dict": model.decoder.state_dict(),
        "selection_head_state_dict": model.selection_head.state_dict(),
        "swept_volume_head_state_dict": model.swept_volume_head.state_dict(),
    }
    
    # Save model config if available
    if hasattr(model, 'config'):
        checkpoint["config"] = model.config
    
    torch.save(checkpoint, save_path)
    print(f"Saved trainable modules to {save_path}")


def load_model_modules(model, load_path, device=None):
    """
    Load only the trainable modules of the model, excluding the neural sweeper.
    
    Args:
        model: The SweepNet model to load into
        load_path: Path to the checkpoint
        device: Device to load the model on
    
    Returns:
        bool: True if loading was successful
    """
    try:
        checkpoint = torch.load(load_path, map_location=device)
        
        if "encoder_state_dict" in checkpoint:
            # Load module-only checkpoint
            model.encoder.load_state_dict(checkpoint["encoder_state_dict"])
            model.decoder.load_state_dict(checkpoint["decoder_state_dict"])
            model.selection_head.load_state_dict(checkpoint["selection_head_state_dict"])
            model.swept_volume_head.load_state_dict(checkpoint["swept_volume_head_state_dict"])
            print(f"Successfully loaded module-only checkpoint from {load_path}")
            return True
        else:
            # Try to load as full model checkpoint (backward compatibility)
            model.load_state_dict(checkpoint["state_dict"])
            print(f"Successfully loaded full model checkpoint from {load_path}")
            return True
            
    except Exception as e:
        print(f"Error loading checkpoint from {load_path}: {e}")
        return False


def get_model_size_info(model):
    """
    Get information about model size and parameter counts.
    
    Args:
        model: The model to analyze
    
    Returns:
        dict: Information about model size
    """
    total_params = 0
    trainable_params = 0
    module_info = {}
    
    for name, module in model.named_modules():
        if hasattr(module, 'parameters'):
            module_params = sum(p.numel() for p in module.parameters())
            module_trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            if module_params > 0:
                module_info[name] = {
                    'total_params': module_params,
                    'trainable_params': module_trainable,
                    'size_mb': module_params * 4 / (1024 * 1024)  # Assuming float32
                }
                total_params += module_params
                trainable_params += module_trainable
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_size_mb': total_params * 4 / (1024 * 1024),
        'trainable_size_mb': trainable_params * 4 / (1024 * 1024),
        'modules': module_info
    }
