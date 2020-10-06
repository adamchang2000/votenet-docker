import open3d as o3d
import numpy as np
from obj_to_pointcloud_util import *


a = np.array([[ 0.20175413,  0.96682634, -0.15665918,],
 [-0.7676377,   0.05675075, -0.63836644,],
 [-0.60829896,  0.24905056,  0.75362205]])

b = np.array([[ 0.20175413, -0.86978975, -0.45028997,],
 [ 0.7676377,   0.42595728, -0.47884522,],
 [ 0.60829896, -0.24905056,  0.75362205]])

print(a @ b)

file = 'D:/model_data_test/3_data.npz'
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

#rotate_matrix_axis_angle(axis_angles, bb)

bb.rotate(eulerAnglesToRotationMatrix(euler_angles), bb.get_center())

print(bb.get_center())
print(bb.get_max_bound() - bb.get_min_bound())

for pc in point_cloud:
	points.append([pc[0], pc[1], pc[2]])
	colors.append([pc[3], pc[4], pc[5]])

points = np.asarray(points)
colors = np.asarray(colors)
pcld.points = o3d.utility.Vector3dVector(points)
pcld.colors = o3d.utility.Vector3dVector(colors)


o3d.visualization.draw_geometries([pcld, bb])
