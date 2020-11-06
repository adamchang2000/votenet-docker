import open3d as o3d
import os, sys
import numpy as np

if (len(sys.argv) != 2):
	print('usage: python reduce_pcld.py <pcld.ply>')
	exit(1)

max_points = 100000
pcld_filename = sys.argv[1]

assert(os.path.exists(pcld_filename))

pcld = o3d.io.read_point_cloud(pcld_filename)
points = np.array(pcld.points)
colors = np.array(pcld.colors)

print(points.shape[0])

index = np.random.choice(points.shape[0], max_points, replace=False)
points = points[index]
colors = colors[index]

pcld.points = o3d.utility.Vector3dVector(points)
pcld.colors = o3d.utility.Vector3dVector(colors)

output_filename = pcld_filename[:-4] + '_sampled.ply'
o3d.io.write_point_cloud(output_filename, pcld)


