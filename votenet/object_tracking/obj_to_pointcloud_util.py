import numpy as np
import open3d as o3d
import argparse
import logging
import math
import cv2

#format = '%(asctime)s.%(msecs)03d %(levelname)-.1s [%(name)s][%(threadName)s] %(message)s'
#logging.basicConfig(format=format, level=logging.DEBUG)

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta):
    
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
        
        
                    
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                    
                    
    R = np.dot(R_z, np.dot( R_y, R_x ))

    return R

def getRandomRotationMatrix():
	euler_angles = np.zeros(3)
	euler_angles[0] = -np.pi + np.random.uniform(0, 1) * 2 * np.pi
	euler_angles[1] = -np.pi / 2 + np.random.uniform(0, 1) * np.pi
	euler_angles[2] = -np.pi + np.random.uniform(0, 1) * 2 * np.pi
	return eulerAnglesToRotationMatrix(euler_angles)

#Get rotation matrix from axis angles.
def axisAnglesToRotationMatrix(axis_angles, theta):

	if theta == 0:
		return np.eye(3)

	assert(abs(np.linalg.norm(axis_angles) - 1) <= 0.0001)
	axis_angles_theta = np.copy(axis_angles) * theta
	return o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles_theta)

def convert_file_to_model(filename, scale = 1):
	try:
		model = o3d.io.read_triangle_mesh(filename)

		#its a pointcloud
		if np.array(model.triangles).shape[0] == 0:
			print('seems like the file is a pointcloud')
			model = o3d.io.read_point_cloud(filename)

		model.scale(scale)
		model.translate(np.asarray([0, 0, 0]), False)
		return model
	except:
		print('load failed xd')
		exit(1)

def place_model(model, xyz, axis_angles, theta):
	#print('place model called')
	rotate_matrix_axis_angle(axis_angles, theta, model)
	model.translate(xyz, relative=False)

def return_origin(model, xyz, axis_angles, theta):
	#print('return origin called')
	model.translate(np.zeros(3), relative=False)
	invert_rotate_matrix_axis_angle(axis_angles, theta, model)

def get_bb(model):
	bb = model.get_axis_aligned_bounding_box()
	return bb

# def rotate_matrix_euler(euler_angles, model):
# 	#print('euler ', eulerAnglesToRotationMatrix(euler_angles))
# 	model.rotate(eulerAnglesToRotationMatrix(euler_angles))
# 	return model

# def invert_rotate_matrix_euler(euler_angles, model):
# 	#print('euler ', eulerAnglesToRotationMatrix(euler_angles))
# 	model.rotate(eulerAnglesToRotationMatrix(euler_angles).T)
# 	return model

def rotate_matrix_axis_angle(axis_angles, theta, model):
	model.rotate(axisAnglesToRotationMatrix(axis_angles, theta))
	return model

def invert_rotate_matrix_axis_angle(axis_angles, theta, model):
	model.rotate(axisAnglesToRotationMatrix(axis_angles, theta).T)
	return model

def get_x_vec(model):
	center = np.asarray(model.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0.01 * i, 0, 0]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

def get_y_vec(model):
	center = np.asarray(model.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0, 0.01 * i, 0]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

def get_z_vec(model):
	center = np.asarray(model.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0, 0, 0.01 * i]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

def sample_points(model, num_points, sample_strategy):

	if type(model) == type(o3d.geometry.TriangleMesh()):
		

		if sample_strategy not in ['uniform random', 'uniform grid']:
			print('%s not uniform random or uniform grid' % sample_strategy)
			sys.exit(1)

		if sample_strategy == 'uniform random':
			pointcloud = model.sample_points_uniformly(num_points)

		elif sample_strategy == 'uniform grid':
			print('dont know if uniform grid works, just exit for now')
			sys.exit(1)
			pointcloud = model.sample_points_poisson_disk(num_points)

	elif type(model) == type(o3d.geometry.PointCloud()):

		if num_points > np.array(model.points).shape[0]:
			lst = np.array([i for i in range(np.array(model.points).shape[0])])
			pointcloud = model.select_down_sample(lst)
		else:
			lst = np.random.choice(np.array(model.points).shape[0], num_points)
			pointcloud = model.select_down_sample(lst)

	else:
		print('only can sample points from trianglemesh or pointcloud, not %s' % repr(type(model)))
		exit(1)

	return pointcloud

#perform augmentations
#add noise to points and color
#delete points in sections?
def augment_pointcloud(pointcloud):

	point_noise_std = 0.01
	color_noise_mean = -0.2
	color_noise_std = 0.1

	color_channel_noise_std = 0.02

	max_holes = 5
	holes_size_mean = 0.02
	holes_size_std = 0.02

	pointcloud.points = o3d.utility.Vector3dVector(np.random.normal(np.array(pointcloud.points), point_noise_std))
	pointcloud.colors = o3d.utility.Vector3dVector(np.array(pointcloud.colors) + np.random.normal(color_noise_mean, color_noise_std))
	pointcloud.colors = o3d.utility.Vector3dVector(np.random.normal(np.array(pointcloud.colors), color_channel_noise_std))

	kd_tree = o3d.geometry.KDTreeFlann(pointcloud)
	points = np.array(pointcloud.points)
	lst = []

	for i in range(np.random.randint(max_holes)):
		point = points[np.random.randint(points.shape[0])]
		v = kd_tree.search_radius_vector_3d(point, max(0, np.random.normal(holes_size_mean, holes_size_std)))[1]
		lst.extend(v)

	pointcloud = pointcloud.select_down_sample(lst, invert=True)

	return pointcloud

#return pointcloud, bounding box, votes, euler_angles
def get_perspective_data_from_model(model, xyz, axis_angles, theta, points=20000, sample_strategy='uniform random'):

	bb = get_bb(model)
	place_model(model, xyz, axis_angles, theta)
	bb.translate(xyz, relative=False)

	#multiply points by 2 for more visible points
	factor = 5
	while True:
		pointcloud = sample_points(model, points * factor, sample_strategy)
		_, lst = pointcloud.hidden_point_removal(np.asarray([0., 0., 0.]), 50)

		if len(lst) < points:
			factor += 5
		elif factor > 20: #less than 5% of the object is showing
			pointcloud = sample_points(model, points, sample_strategy)
			break
		else:
			np.random.shuffle(lst)
			lst = np.array(lst[:points])
			pointcloud = pointcloud.select_down_sample(lst)
			break

	#insert noise into model
	#pointcloud = augment_pointcloud(pointcloud)

	return_origin(model, xyz, axis_angles, theta)

	return pointcloud, bb, axis_angles, theta

#return pointcloud, bb, votes, axis angles
def get_perspective_data_from_model_seed(seed, model, points=20000, sample_strategy='uniform random', center=np.array([0, 0, 0]), scene_scale = 1):
	np.random.seed(seed)

	#print('center ', center)

	#axis_angles = np.random.uniform(-1, 1, size=3)
	#axis_angles /= np.linalg.norm(axis_angles)

	#testing xyz
	axis_angles = np.array([0., 0., 1.])

	theta = np.random.uniform(0, np.pi * 2)

	# euler_angles = np.zeros(3)
	# euler_angles[0] = -np.pi + np.random.uniform(0, 1) * 2 * np.pi
	# euler_angles[1] = -np.pi / 2 + np.random.uniform(0, 1) * np.pi
	# euler_angles[2] = -np.pi + np.random.uniform(0, 1) * 2 * np.pi

	xyz = np.copy(center)
	xyz[0] += np.random.uniform(0.01, 0.1) * (-1 if np.random.rand() < 0.5 else 1) * scene_scale
	xyz[1] += np.random.uniform(0.01, 0.1) * (-1 if np.random.rand() < 0.5 else 1) * scene_scale
	xyz[2] += np.random.uniform(0.01, 0.1) * (-1 if np.random.rand() < 0.5 else 1) * scene_scale

	return get_perspective_data_from_model(model, np.asarray(xyz), axis_angles, theta, points, sample_strategy)

def main():
	parser = argparse.ArgumentParser(description='Extract a model and data from an obj')
	parser.add_argument('--filename', default='medical/textured_medical_patterns_filledin.ply', help='file path to model')
	parser.add_argument('-n', dest='points', default=10000, type=int, help='number of points to sample, default = 20k')
	parser.add_argument('-s', dest='sample_strategy', default='uniform random', help='point sampling strategy, available: uniform random, uniform grid')

	args = parser.parse_args()
	print('starting processing %s' % args.filename)

	model = convert_file_to_model(args.filename, 0.001)
	o3d.visualization.draw_geometries([model])
	
	#pcld, bb, votes, euler_angles = get_perspective_data_from_model(model, np.asarray([1, -3, 2]), np.asarray([0, 0, np.pi / 6]), args.points, args.sample_strategy)

	#o3d.visualization.draw_geometries([pcld, bb])

	pcld, bb, axis_angles, theta = get_perspective_data_from_model_seed(3, model, args.points, args.sample_strategy)

	bb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bb)

	bb.rotate(axisAnglesToRotationMatrix(axis_angles, theta))

	o3d.visualization.draw_geometries([pcld, bb])


if __name__ == "__main__":
	main()