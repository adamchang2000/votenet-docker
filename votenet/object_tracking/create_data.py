import argparse, os, sys
import numpy as np
import open3d as o3d
from obj_to_pointcloud_util import *

def main():
	model_path = 'medical/choukemoduan_v0.obj'
	output_path = 'model_data/'
	scene_path = 'scenes/'
	number_of_samples = 10
	training_number = 8
	testing_number = 2
	val_number = 0
	num_points = 2000
	scale = 0.001

	assert(os.path.exists(model_path))

	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	assert(training_number + testing_number + val_number == number_of_samples)

	model = convert_obj_to_mesh(model_path, scale)

	scenes = []
	for i in range(10):
		scenes.append(o3d.io.read_point_cloud(os.path.join(scene_path, str(i) + '.ply')))

	for i in range(number_of_samples):

		scene_index = np.random.randint(10)
		scene = scenes[scene_index]

		pcld, bb, votes, euler_angles = get_perspective_data_from_mesh_seed(i, model, points=num_points, center = scene.get_center())
		box3d_centers = np.asarray([bb.get_center()])
		box3d_sizes = np.asarray([bb.get_max_bound() - bb.get_min_bound()])

		points = np.asarray(pcld.points)
		colors = np.asarray(pcld.colors)

		combined_points = np.vstack((points, np.array(scene.points)))
		combined_colors = np.vstack((colors, np.array(scene.colors)))
		
		#scene.points = o3d.utility.Vector3dVector(combined_points)
		#scene.colors = o3d.utility.Vector3dVector(combined_colors)

		total_votes = np.zeros((combined_points.shape[0], 3))
		total_votes[:votes.shape[0]] = votes

		#o3d.visualization.draw_geometries([scene])

		assert(len(points) == len(colors))
		assert(len(points) == num_points)

		point_cloud = np.asarray([[p[0], p[1], p[2], c[0], c[1], c[2]] for p,c in zip(combined_points, combined_colors)])

		vote_mask = np.zeros(combined_points.size)
		vote_mask[:points.shape[0]] = 1

		np.savez(output_path + str(i) + '_data.npz', box3d_centers=box3d_centers, box3d_sizes=box3d_sizes, euler_angles=euler_angles, point_cloud=point_cloud, votes=total_votes, vote_mask=vote_mask)

	lst = np.asarray(list(range(number_of_samples)))
	np.random.shuffle(lst)

	np.savez(output_path + 'train_samples.npz', samples=lst[:training_number])
	np.savez(output_path + 'test_samples.npz', samples=lst[training_number:training_number+testing_number])
	np.savez(output_path + 'val_samples.npz', samples=lst[training_number+testing_number:])


if __name__ == "__main__":
	main()