import os
import subprocess
import trimesh
import open3d as o3d
import os
import numpy as np

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

def main(parent_folder):
    for folder in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder)
        print(folder_path)
        if os.path.isdir(folder_path):
            off_file = os.path.join(folder_path, 'model_res_64.off')

            if os.path.exists(off_file):
                # Run the pre-compiled C file
                txt_file = off_file.replace('.off', '.txt')
                if not os.path.exists(txt_file):
                    subprocess.run(['./misc/MCF_Skeleton_example', off_file, txt_file])
                    # Convert .txt to .ply
                    new_ply_file = txt_file.replace(os.path.basename(txt_file), 'skeletal_prior.ply')
                    convert_to_ply(txt_file, new_ply_file)

if __name__ == "__main__":
    parent_folder = './data/gc_objects'  # Replace with the path to the parent folder
    main(parent_folder)
