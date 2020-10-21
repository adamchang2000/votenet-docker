import pywavefront
import numpy as np
import open3d as o3d
import argparse
import logging
import math

pywavefront.configure_logging(
    logging.ERROR
)

#format = '%(asctime)s.%(msecs)03d %(levelname)-.1s [%(name)s][%(threadName)s] %(message)s'
#logging.basicConfig(format=format, level=logging.DEBUG)

# Calculates Rotation Matrix given euler angles.
def eulerAnglesToRotationMatrix(theta) :
    
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

#scale make sure points of vertices are in meters
def convert_obj_to_mesh(filename, scale = 1):

	#print('called convert obj to mesh')

	scene = pywavefront.Wavefront(filename)
	scene.parse()

	vertex_normals = []
	vertex_pos = []
	vertex_colors = []

	for name, material in scene.materials.items():
		i = 0
		vertex = [0, 0]
		while i < len(material.vertices):
			vertex_normals.append([material.vertices[i], material.vertices[i+1], material.vertices[i+2]])
			vertex_pos.append([material.vertices[i+3], material.vertices[i+4], material.vertices[i+5]])
			vertex_colors.append([material.diffuse[0], material.diffuse[1], material.diffuse[2]])
			i += 6

	#print('retrieved data in obj to mesh')

	vertex_normals_np = np.asarray(vertex_normals)
	vertex_pos_np = np.asarray(vertex_pos)

	mesh = o3d.geometry.TriangleMesh()

	triangles = [[3*i, 3*i+1, 3*i+2] for i in range(int(len(vertex_pos) / 3))]

	mesh.triangles = o3d.utility.Vector3iVector(triangles)
	mesh.vertices = o3d.utility.Vector3dVector(vertex_pos)
	mesh.vertex_normals = o3d.utility.Vector3dVector(vertex_normals)
	mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)

	#print('filled mesh in obj to mesh')

	mesh.scale(scale)
	mesh.translate(np.asarray([0, 0, 0]), False)

	#print('finished obj to mesh')

	return mesh

def place_mesh(mesh, xyz, euler_angles):
	#print('place mesh called')
	rotate_matrix_euler(euler_angles, mesh)
	mesh.translate(xyz)

def return_origin(mesh, xyz, euler_angles):
	#print('return origin called')
	mesh.translate(-1 * xyz)
	invert_rotate_matrix_euler(euler_angles, mesh)

def place_mesh_axis_angles(mesh, xyz, axis_angles):
	rotate_matrix_axis_angle(axis_angles, mesh)
	mesh.translate(xyz)

def return_origin_axis_angles(mesh, xyz, axis_angles):
	mesh.translate(-1 * xyz)
	rotate_matrix_axis_angle(-1 * axis_angles, mesh)

def get_bb(mesh):
	bb = mesh.get_axis_aligned_bounding_box()
	return bb

def rotate_matrix_euler(euler_angles, mesh):
	#print('euler ', eulerAnglesToRotationMatrix(euler_angles))
	mesh.rotate(eulerAnglesToRotationMatrix(euler_angles))
	return mesh

def invert_rotate_matrix_euler(euler_angles, mesh):
	#print('euler ', eulerAnglesToRotationMatrix(euler_angles))
	mesh.rotate(eulerAnglesToRotationMatrix(euler_angles).T)
	return mesh

def rotate_matrix_axis_angle(axis_angle, mesh):
	#print('axis angle ', o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle))
	mesh.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angle))
	return mesh

def get_x_vec(mesh):
	center = np.asarray(mesh.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0.01 * i, 0, 0]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

def get_y_vec(mesh):
	center = np.asarray(mesh.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0, 0.01 * i, 0]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

def get_z_vec(mesh):
	center = np.asarray(mesh.get_center())

	lst = []
	for i in range(100):
		lst.append(center + np.asarray([0, 0, 0.01 * i]))

	pcld = o3d.geometry.PointCloud()
	pcld.points = o3d.utility.Vector3dVector(np.asarray(lst))
	return pcld

def sample_points(mesh, num_points, sample_strategy):
	if sample_strategy not in ['uniform random', 'uniform grid']:
		print('%s not uniform random or uniform grid' % sample_strategy)
		sys.exit(1)

	if sample_strategy == 'uniform random':
		pointcloud = mesh.sample_points_uniformly(num_points)

	elif sample_strategy == 'uniform grid':
		print('dont know if uniform grid works, just exit for now')
		sys.exit(1)
		pointcloud = mesh.sample_points_poisson_disk(num_points)

	return pointcloud

#return pointcloud, bounding box, votes, euler_angles
def get_perspective_data_from_mesh(mesh, xyz, euler_angles, points=20000, sample_strategy='uniform random'):

	#print('perspective data called')

	bb = get_bb(mesh)
	place_mesh(mesh, xyz, euler_angles)

	bb.translate(xyz)

	#multiply points by 2 for more visible points
	factor = 5
	while True:
		pointcloud = sample_points(mesh, points * factor, sample_strategy)
		_, lst = pointcloud.hidden_point_removal(np.asarray([0., 0., 0.]), 10000)

		if len(lst) < points:
			factor += 2
		else:
			break

	np.random.shuffle(lst)
	lst = lst[:points]

	pointcloud = pointcloud.select_down_sample(lst)

	votes = []

	for pt in pointcloud.points:
		votes.append(bb.get_center() - pt)

	return_origin(mesh, xyz, euler_angles)

	return pointcloud, bb, np.asarray(votes), euler_angles

#return pointcloud, bounding box, votes, axis_angles
def get_perspective_data_from_mesh_axis_angles(mesh, xyz, axis_angles, points=20000, sample_strategy='uniform random'):

	bb = get_bb(mesh)

	place_mesh_axis_angles(mesh, xyz, axis_angles)

	bb.translate(xyz)

	print('mesh center ', mesh.get_center())
	print('bb center ', bb.get_center())
	
	#multiply points by 2 for more visible points
	factor = 5
	while True:
		pointcloud = sample_points(mesh, points * factor, sample_strategy)

		o3d.visualization.draw_geometries([pcld])

		_, lst = pointcloud.hidden_point_removal(np.asarray([0., 0., 0.]), 10000)

		if len(lst) < points:
			factor += 2
		else:
			break


	np.random.shuffle(lst)
	lst = lst[:points]

	pointcloud = pointcloud.select_down_sample(lst)

	votes = []

	for pt in pointcloud.points:
		votes.append(bb.get_center() - pt)

	return_origin_axis_angles(mesh, xyz, axis_angles)

	return pointcloud, bb, np.asarray(votes), axis_angles

#return pointcloud, bb, votes, axis angles
def get_perspective_data_from_mesh_seed(seed, mesh, points=20000, sample_strategy='uniform random'):
	np.random.seed(seed)
	#axis_angles = np.array([1, 0, 0])
	# axis_angles = np.array([0, 0, 1])
	# axis_angles = axis_angles / np.linalg.norm(axis_angles)
	# axis_angles *= (-np.pi + (np.random.uniform(0, 1) * 2 * np.pi))

	euler_angles = np.zeros(3)
	euler_angles[0] = -np.pi + np.random.uniform(0, 1) * 2 * np.pi
	#euler_angles[1] = -np.pi / 2 + np.random.uniform(0, 1) * np.pi
	#euler_angles[2] = -np.pi + np.random.uniform(0, 1) * 2 * np.pi

	xyz = [0, 0, 0]
	xyz[0] = np.random.uniform(-3, 3)
	xyz[1] = np.random.uniform(0.3, 4)
	xyz[2] = np.random.uniform(-3, 3)

	#return get_perspective_data_from_mesh_axis_angles(mesh, np.asarray(xyz), np.asarray(axis_angles), points, sample_strategy)
	return get_perspective_data_from_mesh(mesh, np.asarray(xyz), np.asarray(euler_angles), points, sample_strategy)

def main():
	parser = argparse.ArgumentParser(description='Extract a mesh and data from an obj')
	parser.add_argument('--filename', default='ps3_controller/model.obj', help='file path to obj')
	parser.add_argument('-n', dest='points', default=10000, type=int, help='number of points to sample, default = 20k')
	parser.add_argument('-s', dest='sample_strategy', default='uniform random', help='point sampling strategy, available: uniform random, uniform grid')

	args = parser.parse_args()
	print('starting processing %s' % args.filename)

	mesh = convert_obj_to_mesh(args.filename, 0.001)
	
	#pcld, bb, votes, euler_angles = get_perspective_data_from_mesh(mesh, np.asarray([1, -3, 2]), np.asarray([0, 0, np.pi / 6]), args.points, args.sample_strategy)

	#o3d.visualization.draw_geometries([pcld, bb])

	pcld, bb, votes, euler_angles = get_perspective_data_from_mesh_seed(3, mesh, args.points, args.sample_strategy)

	print(bb.get_max_bound())
	print(bb.get_min_bound())

	bb = o3d.geometry.OrientedBoundingBox.create_from_axis_aligned_bounding_box(bb)

	bb.rotate(eulerAnglesToRotationMatrix(euler_angles))

	o3d.visualization.draw_geometries([pcld, bb])


if __name__ == "__main__":
	main()