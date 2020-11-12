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
	num_points = 1000
	scale = 0.001

	assert(os.path.exists(model_path))

	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	assert(training_number + testing_number + val_number == number_of_samples)

	model = convert_obj_to_mesh(model_path, scale)

	scenes = []
	for i in range(123):
		scenes.append(o3d.io.read_point_cloud(os.path.join(scene_path, str(i) + '.ply')))
		#scenes.append('xd')

	for i in range(number_of_samples):

		if i % 100 == 0:
			print('creating sample ', i + 1, end='\r')

		scene_index = np.random.randint(123)
		scene = scenes[scene_index]

		scene_pts = np.array(scene.points)

		pcld, bb, votes, euler_angles = get_perspective_data_from_mesh_seed(i, model, points=num_points, center = scene_pts[np.random.randint(scene_pts.shape[0])])
		box3d_centers = np.asarray([bb.get_center()])
		box3d_sizes = np.asarray([bb.get_max_bound() - bb.get_min_bound()])

		# pts = np.array(pcld.points)
		# print(pts.shape)
		# colors = np.array(pcld.colors)
		# print(colors[3])
		# print(colors[4])

		# o3d.visualization.draw_geometries([pcld, bb])

		# exit()

		points = np.asarray(pcld.points)
		colors = np.asarray(pcld.colors)

		combined_points = np.vstack((points, np.array(scene.points)))
		combined_colors = np.vstack((colors, np.array(scene.colors)))

		total_votes = np.zeros((combined_points.shape[0], 3))
		total_votes[:votes.shape[0]] = votes

		assert(len(combined_points) == len(combined_colors))

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