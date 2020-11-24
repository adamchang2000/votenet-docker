import argparse, os, sys
import numpy as np
import open3d as o3d
from obj_to_pointcloud_util import *

def main():
	model_path = 'medical/new_textured_medical_patterns.ply'
	output_path = 'model_data/'
	scene_path = 'scenes_2/'
	number_of_samples = 100
	training_number = 80
	testing_number = 20
	val_number = 0
	num_points = 4000
	scale_output_pcld = 0.3
	scale = 0.001 * scale_output_pcld
	

	assert(os.path.exists(model_path))

	if not os.path.isdir(output_path):
		os.mkdir(output_path)

	assert(training_number + testing_number + val_number == number_of_samples)

	model = convert_file_to_model(model_path, scale)
	# print(model.get_max_bound())
	# print(model.get_min_bound())
	o3d.visualization.draw_geometries([model])

	scenes = []
	for i in range(52):
		scenes.append(o3d.io.read_point_cloud(os.path.join(scene_path, str(i) + '.ply')))

	for i in range(number_of_samples):

		if i % 50 == 1:
			print('creating sample ', i, end='\r')

		scene_index = np.random.randint(52)
		scene = scenes[scene_index]

		scene_pts = np.array(scene.points) * scale_output_pcld

		pcld, bb, votes, euler_angles = get_perspective_data_from_model_seed(i, model, points=num_points, center = scene_pts[np.random.randint(scene_pts.shape[0])], scene_scale = scale_output_pcld)

		#o3d.visualization.draw_geometries([pcld, bb])

		box3d_centers = np.asarray([bb.get_center()])
		box3d_sizes = np.asarray([bb.get_max_bound() - bb.get_min_bound()])

		points = np.asarray(pcld.points)
		colors = np.asarray(pcld.colors)

		combined_points = np.vstack((points, np.array(scene_pts))) 
		combined_colors = np.vstack((colors, np.array(scene.colors)))

		pcld_out = o3d.geometry.PointCloud()
		pcld_out.points = o3d.utility.Vector3dVector(combined_points)
		#pcld_out.colors = o3d.utility.Vector3dVector(colors)
		
		#o3d.visualization.draw_geometries([pcld_out, bb])
		#o3d.io.write_point_cloud(str(i) + '.ply', pcld_out)

		#exit()

		total_votes = np.zeros((combined_points.shape[0], 3))
		total_votes[:votes.shape[0]] = votes

		assert(len(combined_points) == len(combined_colors))

		#point_cloud = np.asarray([[p[0], p[1], p[2], c[0], c[1], c[2]] for p,c in zip(combined_points, combined_colors)])

		#single channel, 1 or 0, after adaptive threshold filter
		point_cloud = np.asarray([[p[0], p[1], p[2], c[0]] for p,c in zip(combined_points, combined_colors)])

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