import open3d as o3d
import numpy as np
import sys, os, math

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

def calculate_labels(pcld, bb, angles, file):
	object_pcld = pcld.crop(bb)
	#o3d.visualization.draw_geometries([object_pcld])

	object_pts = [(pt[0], pt[1], pt[2]) for pt in np.array(object_pcld.points)]
	object_set = set(object_pts)

	pcld_pts = [((p[0], p[1], p[2]), c[0]) for p, c in zip(np.array(pcld.points), np.array(pcld.colors))]


	bb_center = np.array(bb.get_center())

	votes = []
	vote_mask = []

	for point, color in pcld_pts:
		if point in object_set:
			votes.append(np.subtract(bb_center, point))
			vote_mask.append(1)
		else:
			votes.append(np.zeros(3))
			vote_mask.append(0)

	box3d_centers = np.asarray([bb.get_center()])
	box3d_sizes = np.asarray([bb.extent])

	point_cloud = np.array([(p[0], p[1], p[2], c) for p, c in pcld_pts])

	total_votes = np.array(votes)
	vote_mask = np.array(vote_mask)

	np.savez(file[:-4] + '_data.npz', box3d_centers=box3d_centers, box3d_sizes=box3d_sizes, euler_angles=angles, point_cloud=point_cloud, votes=total_votes, vote_mask=vote_mask)

def write_val_samples(file):
	directory = os.path.dirname(os.path.realpath(file))
	files = [a for a in os.listdir(directory) if '_data.npz' in a]
	files = [int(a[:a.find('_data')]) for a in files]
	np.savez(os.path.join(directory, 'val_samples.npz'), samples=files)

#this training data will be rotation ambiguous, but the important matter is the xyz voting
def main():
	file = sys.argv[1]
	assert(os.path.exists(file))

	pcld = o3d.io.read_point_cloud(file)
	bb = o3d.geometry.OrientedBoundingBox()

	print('pointcloud points: ', pcld.get_max_bound(), pcld.get_min_bound())

	#x - right, y - down, z - away from camera
	xyz = np.array([-0.01, 0.06, 0.52])
	#first: -pi, pi
	#second: -pi/2, pi/2
	#third: -pi, pi
	rotate = [-3.,  -0.8, 1.5]
	extent = np.array([0.08954177, 0.125, 0.14596413])
	scale = 1.

	bb.translate(xyz, relative=False)
	bb.extent = extent
	bb.scale(scale)

	happy = False

	if happy:
		bb.rotate(eulerAnglesToRotationMatrix(rotate))
		calculate_labels(pcld, bb, rotate, file)
		write_val_samples(file)

	else:

		testing = False

		if testing:
			for i in range(4):
				for k in range(4):
					for j in range(4):
						test_rot = (-np.pi + 1/2 * np.pi * i, -np.pi/2 + 1/4 * np.pi * k, -np.pi + 1/2 * np.pi * j)
						bb.rotate(eulerAnglesToRotationMatrix(test_rot))

						print(i * 16 + k * 4 + j, test_rot)

						o3d.visualization.draw_geometries([pcld, bb])

						bb.rotate(np.linalg.inv(eulerAnglesToRotationMatrix(test_rot)))

						
		else:
			bb.rotate(eulerAnglesToRotationMatrix(rotate))
			o3d.visualization.draw_geometries([pcld, bb])
	

	


if __name__ == '__main__':
	main()




