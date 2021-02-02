import argparse, os, sys
import numpy as np
import open3d as o3d
from obj_to_pointcloud_util import *
from datetime import datetime

def main():
	model_path = 'medical/medical_bulbs.ply'
	#model_path = 'medical/medical_textured.obj'
	output_path = 'model_data/'
	scene_path = 'scenes_highdens/'
	number_of_samples = 100
	training_number = 80
	testing_number = 15
	val_number = 5
	num_points = 100000
	scale_output_pcld = 1.0
	scale = 0.001 * scale_output_pcld
	

	assert(os.path.exists(model_path))

	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	assert(training_number + testing_number + val_number == number_of_samples)

	model = convert_file_to_model(model_path, scale)
	# print(model.get_max_bound())
	# # print(model.get_min_bound())
	#o3d.visualization.draw_geometries([model])
	#if np.unique(np.array(model.colors)).shape[0] != 2:
	#	print('WARNING, COLORS ARE NOT BINARY')
	#print(np.unique(np.array(model.colors)))
	# exit()

	scenes = []

	for scene_file in os.listdir(scene_path):
		scenes.append(o3d.io.read_point_cloud(os.path.join(scene_path, scene_file)))

	i = 0
	while i < number_of_samples:

		if i % 50 == 0:
			print('creating sample ', i, datetime.now(), end='\r')

		scene_index = np.random.randint(len(scenes))
		scene = scenes[scene_index]

		scene_pts = np.array(scene.points) * scale_output_pcld
		scene_colors = np.array(scene.colors)

		center = scene_pts[np.random.randint(scene_pts.shape[0])]

		model_pcld, bb, axis_angles, theta = get_perspective_data_from_model_seed(np.random.randint(99999999), model, points=num_points, center = center, scene_scale = scale_output_pcld)

		box3d_centers = np.asarray([bb.get_center()])
		box3d_sizes = np.asarray([bb.get_max_bound() - bb.get_min_bound()])

		model_points = np.asarray(model_pcld.points)
		model_colors = np.asarray(model_pcld.colors)

		# combined_points = np.vstack((points, np.array(scene_pts))) 
		# combined_colors = np.vstack((colors, np.array(scene.colors)))

		# total_votes = np.zeros((combined_points.shape[0], 3))
		# total_votes[:votes.shape[0]] = votes

		#single channel, 1 or 0, after adaptive threshold filter
		scene_point_cloud = np.asarray([[p[0], p[1], p[2], 0] for p,c in zip(scene_pts, scene_colors)])

		scene_point_cloud = scene_point_cloud[np.sum(np.square(scene_pts - center), axis = 1) < 0.05]

		#print(scene_point_cloud.shape)

		#a = np.sum(np.square(scene_pts - center), axis = 1)
		#print(a)

		#scene_point_cloud = np.array([[0, 0, 0, 0]])
		model_point_cloud = np.asarray([[p[0], p[1], p[2], c[0]] for p,c in zip(model_points, model_colors)])

		# vote_mask = np.zeros(combined_points.size)
		# vote_mask[:points.shape[0]] = 1

		print(scene_point_cloud.shape[0], model_point_cloud.shape[0])

		if (scene_point_cloud.shape[0] + model_point_cloud.shape[0] > 25000):
			np.savez(output_path + str(i) + '_data.npz', box3d_centers=box3d_centers, axis_angles=axis_angles, theta=theta, scene_point_cloud=scene_point_cloud, model_point_cloud=model_point_cloud)
			i += 1


	lst = np.asarray(list(range(number_of_samples)))
	np.random.shuffle(lst)

	np.savez(output_path + 'train_samples.npz', samples=lst[:training_number])
	np.savez(output_path + 'test_samples.npz', samples=lst[training_number:training_number+testing_number])
	np.savez(output_path + 'val_samples.npz', samples=lst[training_number+testing_number:])


if __name__ == "__main__":
	main()