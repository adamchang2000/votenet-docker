import argparse, os, sys
import numpy as np
from obj_to_pointcloud_util import *

def main():
	model_path = 'ps3_controller/model.obj'
	output_path = 'D:/model_data_aa/'
	number_of_samples = 100
	training_number = 80
	testing_number = 20
	val_number = 0
	num_points = 10000
	scale = 0.001

	assert(os.path.exists(model_path))

	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	assert(training_number + testing_number + val_number == number_of_samples)

	model = convert_obj_to_mesh(model_path, scale)

	noise = 0

	for i in range(number_of_samples):
		pcld, bb, votes, axis_angles = get_perspective_data_from_mesh_seed(i, model, points=num_points-noise)
		box3d_centers = np.asarray([bb.get_center()])
		box3d_sizes = np.asarray([bb.get_max_bound() - bb.get_min_bound()])

		points = np.asarray(pcld.points)
		colors = np.asarray(pcld.colors)

		#print('before ', points.shape, colors.shape)

		points = np.vstack((points, np.random.uniform(-2, 2, (noise, 3))))
		colors = np.vstack((colors, np.random.uniform(-1, 1, (noise, 3))))

		#print('after ', points.shape, colors.shape)

		assert(len(points) == len(colors))
		assert(len(points) == num_points)

		point_cloud = np.asarray([[p[0], p[1], p[2], c[0], c[1], c[2]] for p,c in zip(points,colors)])

		np.savez(output_path + str(i) + '_data.npz', box3d_centers=box3d_centers, box3d_sizes=box3d_sizes, axis_angles=axis_angles, point_cloud=point_cloud, votes=votes)

	lst = np.asarray(list(range(number_of_samples)))
	np.random.shuffle(lst)

	np.savez(output_path + 'train_samples.npz', samples=lst[:training_number])
	np.savez(output_path + 'test_samples.npz', samples=lst[training_number:training_number+testing_number])
	np.savez(output_path + 'val_samples.npz', samples=lst[training_number+testing_number:])


if __name__ == "__main__":
	main()