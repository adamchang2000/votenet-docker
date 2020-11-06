import open3d as o3d
import os, sys

if (len(sys.argv) != 2):
	print('usage: python convert_mesh_to_pcld.py <mesh.ply>')
	exit(1)

max_points = 100000
mesh_filename = sys.argv[1]

assert(os.path.exists(mesh_filename))

mesh = o3d.io.read_triangle_mesh(mesh_filename)

pcld = mesh.sample_points_poisson_disk(max_points)

output_filename = mesh_filename[:-4] + "_pcld.ply"
o3d.io.write_point_cloud(output_filename, pcld)