import open3d as o3d
import numpy as np
from obj_to_pointcloud_util import *
from scipy.spatial.transform import Rotation as R

file = 'model_data/4_data.npz'
data = np.load(file)

pcld = o3d.geometry.PointCloud()
scene_point_cloud = data['scene_point_cloud']
model_point_cloud = data['model_point_cloud']
points = []
colors = []

box3d_centers = data['box3d_centers']
axis_angles = data['axis_angles']
theta = data['theta']

rot = R.from_rotvec(axis_angles * theta)

new_axis_angles = np.random.uniform(-1, 1, size=3)
new_axis_angles /= np.linalg.norm(new_axis_angles)

new_theta = np.random.uniform(0, np.pi * 2)

model_point_cloud_centered = model_point_cloud[:,:3] - box3d_centers
model_point_cloud[:,:3] = (axisAnglesToRotationMatrix(new_axis_angles, new_theta) @ model_point_cloud_centered.T).T + box3d_centers

rot2 = R.from_rotvec(new_axis_angles * new_theta)

composed_axis_angles = (rot2 * rot).as_rotvec()
composed_theta = np.linalg.norm(composed_axis_angles)
composed_axis_angles /= np.linalg.norm(composed_axis_angles)

bb = o3d.geometry.OrientedBoundingBox()
bb.center = box3d_centers[0]
bb.extent = np.array([0.08951334, 0.125, 0.14595903])

print(bb.get_center())
print(bb.get_max_bound() - bb.get_min_bound())

print("\n ---------------------------- \n")

#rotate_matrix_axis_angle(axis_angles, bb)

bb.rotate(axisAnglesToRotationMatrix(composed_axis_angles, composed_theta))

print(bb.get_center())

print("?")

print(bb.get_max_bound())
print(bb.get_min_bound())

for pc in scene_point_cloud:
	points.append([pc[0], pc[1], pc[2]])
	colors.append([pc[3], pc[3], pc[3]])

for pc in model_point_cloud:
	points.append([pc[0], pc[1], pc[2]])
	colors.append([pc[3], pc[3], pc[3]])

points = np.asarray(points)
colors = np.asarray(colors)

print(np.unique(colors))

pcld.points = o3d.utility.Vector3dVector(points)
#pcld.colors = o3d.utility.Vector3dVector(colors)


print('pc bounds:')
print(pcld.get_max_bound())
print(pcld.get_min_bound())

o3d.visualization.draw_geometries([pcld, bb])
