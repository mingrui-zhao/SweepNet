import os
import subprocess
import trimesh
import open3d as o3d
import os
import numpy as np
import json
import branch_utils


def read_polyline(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    segments = []
    for line in lines:
        parts = line.split()
        if len(parts) == 7 and parts[0] == '2':
            x1, y1, z1, x2, y2, z2 = map(float, parts[1:])
            segments.append([(x1, y1, z1), (x2, y2, z2)])
    return segments

def convert_to_ply(file_path, output_path):
    segments = read_polyline(file_path)
    points = [point for segment in segments for point in segment]
    points = np.unique(np.array(points), axis=0)  # Remove duplicate points

    # Creating Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Saving the point cloud as a PLY file
    o3d.io.write_point_cloud(output_path, pcd)

def process_skeleton_branches(txt_file, output_dir):
    """
    Process skeleton to extract branch information and save it.
    """
    # Read skeleton segments
    segments = branch_utils.read_skeleton_segments(txt_file)
    
    # Extract branches
    branches = branch_utils.extract_skeleton_branches(segments)
    
    # Save branch information as JSON
    branch_info = {
        'branches': [
            {
                'id': i,
                'points': branch,
                'length': branch_utils.calculate_branch_length(branch)
            }
            for i, branch in enumerate(branches)
        ],
        'total_branches': len(branches),
        'segments': segments
    }
    
    branch_info_path = os.path.join(output_dir, 'branch_info.json')
    with open(branch_info_path, 'w') as f:
        json.dump(branch_info, f, indent=2)
    
    return branches, branch_info

def main(parent_folder):
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        print(f"Processing {folder_path}")
        if os.path.isdir(folder_path):
            off_file = os.path.join(folder_path, 'voxel_64_mc.off')

            if os.path.exists(off_file):
                # Run the pre-compiled C file
                txt_file = off_file.replace('.off', '.txt')
                if not os.path.exists(txt_file):
                    print(f"  Generating skeleton for {off_file}")
                    try:
                        subprocess.run(['./misc/MCF_Skeleton_example', off_file, txt_file], check=True)
                    except subprocess.CalledProcessError as e:
                        print(f"  Error generating skeleton: {e}")
                        continue
                
                # Process branch information
                print(f"  Processing branch information...")
                try:
                    branches, branch_info = process_skeleton_branches(txt_file, folder_path)
                    print(f"    Found {len(branches)} branches")
                    
                    # Save individual branch point clouds
                    for i, branch in enumerate(branches):
                        if len(branch) > 1:
                            branch_points = np.array(branch)
                            branch_pcd = o3d.geometry.PointCloud()
                            branch_pcd.points = o3d.utility.Vector3dVector(branch_points)
                            branch_ply_path = os.path.join(folder_path, f'branch_{i}.ply')
                            o3d.io.write_point_cloud(branch_ply_path, branch_pcd)
                    
                except Exception as e:
                    print(f"  Error processing branches: {e}")
                    branches = []
                
                # Convert .txt to .ply (original functionality)
                new_ply_file = txt_file.replace(os.path.basename(txt_file), 'skeletal_prior.ply')
                if not os.path.exists(new_ply_file):
                    print(f"  Converting to PLY format...")
                    convert_to_ply(txt_file, new_ply_file)
                
                # Also create mcf_skeleton.ply for compatibility with SweepDataPCD
                mcf_skeleton_file = os.path.join(folder_path, 'mcf_skeleton.ply')
                if not os.path.exists(mcf_skeleton_file):
                    convert_to_ply(txt_file, mcf_skeleton_file)
                
                print(f"  Completed processing {folder}")

if __name__ == "__main__":
    parent_folder = './data/GC_objects'  # Replace with the path to the parent folder
    main(parent_folder)
