"""Branch extraction utilities for skeletal structures."""

import numpy as np
import torch
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any
import os


def read_skeleton_segments(file_path: str) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Read skeleton segments from MCF skeleton output file.
    Returns list of line segments as [(x1,y1,z1), (x2,y2,z2)] pairs.
    """
    segments = []
    if not os.path.exists(file_path):
        return segments
        
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 7 and parts[0] == '2':
            x1, y1, z1, x2, y2, z2 = map(float, parts[1:])
            segments.append(((x1, y1, z1), (x2, y2, z2)))
    
    return segments


def build_skeleton_graph(segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]) -> Dict[Tuple[float, float, float], List[Tuple[float, float, float]]]:
    """
    Build adjacency graph from skeleton segments.
    Returns dict where keys are points and values are lists of connected points.
    """
    graph = defaultdict(list)
    
    for seg in segments:
        p1, p2 = seg
        graph[p1].append(p2)
        graph[p2].append(p1)
    
    return graph


def find_junction_and_endpoint_nodes(graph: Dict[Tuple[float, float, float], List[Tuple[float, float, float]]]) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:
    """
    Find junction points (degree > 2) and endpoints (degree == 1) in skeleton graph.
    Returns (junctions, endpoints).
    """
    junctions = []
    endpoints = []
    
    for point, neighbors in graph.items():
        degree = len(neighbors)
        if degree > 2:
            junctions.append(point)
        elif degree == 1:
            endpoints.append(point)
    
    return junctions, endpoints


def extract_path_between_points(start: Tuple[float, float, float], 
                               end: Tuple[float, float, float], 
                               graph: Dict[Tuple[float, float, float], List[Tuple[float, float, float]]]) -> List[Tuple[float, float, float]]:
    """
    Extract path between two points using BFS.
    Returns list of points forming the path.
    """
    if start == end:
        return [start]
    
    queue = deque([(start, [start])])
    visited = {start}
    
    while queue:
        current, path = queue.popleft()
        
        for neighbor in graph[current]:
            if neighbor == end:
                return path + [neighbor]
            
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))
    
    return []  # No path found


def extract_branch_from_endpoint(endpoint: Tuple[float, float, float], 
                                graph: Dict[Tuple[float, float, float], List[Tuple[float, float, float]]], 
                                junctions: List[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """
    Extract branch starting from an endpoint until reaching a junction.
    Returns list of points forming the branch.
    """
    branch = [endpoint]
    current = endpoint
    visited = {endpoint}
    
    while True:
        neighbors = [n for n in graph[current] if n not in visited]
        
        if not neighbors:
            break
            
        next_point = neighbors[0]  # Follow the only unvisited neighbor
        branch.append(next_point)
        visited.add(next_point)
        current = next_point
        
        # Stop if we reach a junction
        if current in junctions:
            break
    
    return branch


def calculate_branch_length(branch: List[Tuple[float, float, float]]) -> float:
    """
    Calculate the total length of a branch.
    """
    length = 0.0
    for i in range(len(branch) - 1):
        p1 = np.array(branch[i])
        p2 = np.array(branch[i + 1])
        length += np.linalg.norm(p2 - p1)
    return length


def extract_skeleton_branches(segments: List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]) -> List[List[Tuple[float, float, float]]]:
    """
    Extract individual branches from skeleton line segments.
    Returns list of branches, where each branch is a list of points.
    """
    if not segments:
        return []
    
    # Build adjacency graph
    graph = build_skeleton_graph(segments)
    
    # Find junction points and endpoints
    junctions, endpoints = find_junction_and_endpoint_nodes(graph)
    
    # Extract branches from endpoints to junctions
    branches = []
    for endpoint in endpoints:
        branch = extract_branch_from_endpoint(endpoint, graph, junctions)
        if len(branch) > 1:  # Only keep branches with more than 1 point
            branches.append(branch)
    
    # If no junctions exist (simple chain), treat whole skeleton as one branch
    if not junctions and len(endpoints) <= 2:
        # Find the longest path in the graph
        if endpoints:
            start = endpoints[0]
            end = endpoints[1] if len(endpoints) > 1 else endpoints[0]
            main_branch = extract_path_between_points(start, end, graph)
            if main_branch:
                branches.append(main_branch)
    
    return branches


def fit_bspline_to_branch(branch: List[Tuple[float, float, float]], num_control_points: int = 3) -> np.ndarray:
    """
    Fit B-spline control points to a branch.
    Returns control points as numpy array of shape (num_control_points, 3).
    """
    if len(branch) < num_control_points:
        # If branch is too short, just repeat points
        branch_array = np.array(branch)
        indices = np.linspace(0, len(branch) - 1, num_control_points).astype(int)
        return branch_array[indices]
    
    # Convert branch to numpy array
    branch_array = np.array(branch)
    
    # Sample control points evenly along the branch
    indices = np.linspace(0, len(branch) - 1, num_control_points).astype(int)
    control_points = branch_array[indices]
    
    return control_points


def assign_primitives_to_branches(branches: List[List[Tuple[float, float, float]]], 
                                 num_primitives: int,
                                 bspline_control_points: int = 3) -> Tuple[List[np.ndarray], List[int]]:
    """
    Assign primitives to branches based on their relative lengths with balanced distribution.
    Returns (primitive_control_points, branch_assignments).
    """
    if not branches:
        # Fallback: create random control points
        control_points = []
        for _ in range(num_primitives):
            control_points.append(np.random.randn(bspline_control_points, 3).astype(np.float32))
        return control_points, list(range(num_primitives))
    
    # Calculate branch lengths
    branch_lengths = [calculate_branch_length(branch) for branch in branches]
    total_length = sum(branch_lengths)
    
    if total_length == 0:
        # Fallback for zero-length branches
        control_points = []
        branch_assignments = []
        for i in range(num_primitives):
            if i < len(branches):
                control_points.append(fit_bspline_to_branch(branches[i], bspline_control_points))
                branch_assignments.append(i)
            else:
                control_points.append(np.random.randn(bspline_control_points, 3).astype(np.float32))
                branch_assignments.append(0)
        return control_points, branch_assignments
    
    # Improved assignment strategy: balanced distribution with length weighting
    primitive_control_points = []
    branch_assignments = []
    
    # Calculate base assignment: each branch gets at least one primitive
    base_assignments = min(len(branches), num_primitives)
    remaining_primitives = num_primitives - base_assignments
    
    # First pass: assign one primitive to each branch
    for i, branch in enumerate(branches[:base_assignments]):
        control_points = fit_bspline_to_branch(branch, bspline_control_points)
        primitive_control_points.append(control_points)
        branch_assignments.append(i)
    
    # Second pass: distribute remaining primitives based on branch lengths
    if remaining_primitives > 0:
        # Calculate relative weights (longer branches get more primitives)
        weights = np.array(branch_lengths, dtype=np.float64) / float(total_length)
        
        # Distribute remaining primitives proportionally
        additional_per_branch = np.round(weights * float(remaining_primitives)).astype(np.int32)
        
        # Adjust if we have too many or too few total assignments
        total_additional = int(np.sum(additional_per_branch))
        if total_additional > remaining_primitives:
            # Remove excess from longest branches
            excess = total_additional - remaining_primitives
            sorted_indices = np.argsort(branch_lengths)[::-1]  # Longest first
            for i in range(excess):
                branch_idx = int(sorted_indices[i % len(sorted_indices)])
                if additional_per_branch[branch_idx] > 0:
                    additional_per_branch[branch_idx] -= 1
        elif total_additional < remaining_primitives:
            # Add deficit to longest branches
            deficit = remaining_primitives - total_additional
            sorted_indices = np.argsort(branch_lengths)[::-1]  # Longest first
            for i in range(deficit):
                branch_idx = int(sorted_indices[i % len(sorted_indices)])
                additional_per_branch[branch_idx] += 1
        
        # Add additional primitives to branches
        for branch_idx, additional_count in enumerate(additional_per_branch):
            additional_count = int(additional_count)
            if additional_count > 0 and branch_idx < len(branches):
                branch = branches[branch_idx]
                
                # Create diverse primitives along the branch
                for j in range(additional_count):
                    if len(primitive_control_points) >= num_primitives:
                        break
                    
                    # Create segments for multiple primitives on the same branch
                    total_segments = additional_count + 1  # +1 for the base primitive
                    segment_idx = j + 1  # Skip first segment (already assigned)
                    
                    # Calculate segment boundaries
                    segment_start = int((segment_idx * (len(branch) - 1)) / total_segments)
                    segment_end = int(((segment_idx + 1) * (len(branch) - 1)) / total_segments) + 1
                    
                    # Ensure we don't go out of bounds
                    segment_start = max(0, min(segment_start, len(branch) - 1))
                    segment_end = max(segment_start + 1, min(segment_end, len(branch)))
                    
                    branch_segment = branch[segment_start:segment_end]
                    
                    # Add some variation to prevent identical primitives
                    if len(branch_segment) >= bspline_control_points:
                        control_points = fit_bspline_to_branch(branch_segment, bspline_control_points)
                    else:
                        # For short segments, use the whole branch with small perturbation
                        control_points = fit_bspline_to_branch(branch, bspline_control_points)
                        # Add small random perturbation to avoid identical primitives
                        perturbation = np.random.normal(0, 0.02, control_points.shape).astype(np.float32)
                        control_points = control_points.astype(np.float32) + perturbation
                    
                    primitive_control_points.append(control_points)
                    branch_assignments.append(branch_idx)
    
    # Final check: ensure we have exactly num_primitives
    while len(primitive_control_points) < num_primitives:
        # Add to the longest branch
        longest_branch_idx = int(np.argmax(branch_lengths))
        longest_branch = branches[longest_branch_idx]
        control_points = fit_bspline_to_branch(longest_branch, bspline_control_points)
        # Add perturbation to avoid identical primitives
        perturbation = np.random.normal(0, 0.02, control_points.shape).astype(np.float32)
        control_points = control_points.astype(np.float32) + perturbation
        primitive_control_points.append(control_points)
        branch_assignments.append(longest_branch_idx)
    
    # Trim if we have too many
    primitive_control_points = primitive_control_points[:num_primitives]
    branch_assignments = branch_assignments[:num_primitives]
    
    return primitive_control_points, branch_assignments


def process_skeleton_for_branch_initialization(skeleton_txt_path: str, 
                                             num_primitives: int,
                                             bspline_control_points: int = 3) -> Tuple[List[np.ndarray], List[int], List[List[Tuple[float, float, float]]]]:
    """
    Process skeleton file for branch-wise initialization.
    Returns (primitive_control_points, branch_assignments, branches).
    """
    # Read skeleton segments
    segments = read_skeleton_segments(skeleton_txt_path)
    
    # Extract branches
    branches = extract_skeleton_branches(segments)
    
    # Assign primitives to branches
    primitive_control_points, branch_assignments = assign_primitives_to_branches(
        branches, num_primitives, bspline_control_points
    )
    
    return primitive_control_points, branch_assignments, branches 