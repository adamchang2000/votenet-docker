import open3d as o3d
import numpy as np
from obj_to_pointcloud_util import *

file = 'D:/model_data2/50_data.npz'
data = np.load(file)

pcld = o3d.geometry.PointCloud()
point_cloud = data['point_cloud']
points = []
colors = []

print(len(point_cloud))

bb_center = data['box3d_centers']
bb_size = data['box3d_sizes']
euler_angles = data['euler_angles']

bb = o3d.geometry.OrientedBoundingBox()
bb.center = bb_center[0]
bb.extent = bb_size[0]

print(bb.get_center())
print(bb.get_max_bound() - bb.get_min_bound())

print("\n ---------------------------- \n")

rotate_matrix_euler(euler_angles, bb)

print(bb.get_center())
print(bb.get_max_bound() - bb.get_min_bound())

for pc in point_cloud:
	points.append([pc[0], pc[1], pc[2]])
	colors.append([pc[3], pc[4], pc[5]])

points = np.asarray(points)
colors = np.asarray(colors)
pcld.points = o3d.utility.Vector3dVector(points)
pcld.colors = o3d.utility.Vector3dVector(colors)

pcld.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1,1,0])), pcld.get_center())


o3d.visualization.draw_geometries([pcld, bb])
