import open3d as o3d
from scipy import ndimage
import cv2
import numpy as np
import inspect
from skimage.color import rgb2hsv

from datetime import datetime

def color2grayscale(image):
	return np.dot(image, [0.2989, 0.5870, 0.1140])


#run a sobel filter on 2d image
#input: np array, 3 channel, 0-1
def sobel_filter_2d(image):

	image = color2grayscale(image)

	cv2.imshow('test_image', image)
	cv2.waitKey(0)

	sobel_kernel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
	sobel_kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

	sobel_image_x = ndimage.convolve(image, sobel_kernel_x, mode='constant', cval=0.0)
	sobel_image_y = ndimage.convolve(image, sobel_kernel_y, mode='constant', cval=0.0)

	sobel_image = np.sqrt(np.square(sobel_image_x) + np.square(sobel_image_y))

	cv2.imshow('test', sobel_image)
	cv2.waitKey(0)

	return sobel_image


#run a sobel filter on a 3d mesh
#input: mesh o3d.geometry.TriangleMesh
#output: mesh o3d.geometry.TriangleMesh with vertex_colors sobelized
def sobel_filter_3d_mesh(mesh):
	#create copy
	output_mesh = o3d.geometry.TriangleMesh(mesh)
	if not output_mesh.has_vertex_normals():
		output_mesh.compute_vertex_normals()
	vertices = np.array(output_mesh.vertices)
	normals = np.array(output_mesh.vertex_normals)

	#o3d.visualization.draw_geometries([output_mesh])

	#construct kd tree from mesh
	kd_tree = o3d.geometry.KDTreeFlann(output_mesh)

	for point, normal, i in zip(vertices, normals, range(vertices.shape[0])):
		v = kd_tree.search_radius_vector_3d(point, 0.001)
		verts = vertices[v[1]]
		vecs = verts - point

		print(vecs.shape)

		temp = np.sum(vecs, axis=1) > 0
		verts = verts[temp]
		vecs = vecs[temp]

		print(vecs.shape)

		print('point ', point)
		print('vecs ', vecs)

		normal_normd = normal - point
		normal_normd /= np.linalg.norm(normal_normd)

		a = np.linalg.norm(vecs, axis=1)
		vecs = vecs / a[:,None]

		print('normal, normd ', normal_normd)

		for vec in vecs:
			print(vec)
			print(np.dot(vec, normal_normd))

		print(vecs.shape)
		print(normal_normd.shape)
		#verts = verts[np.dot(vecs, normal_normd) < 0]

		print('yikes')
		print(normal_normd)
		print(verts)
		exit()

	print(kd_tree)

	pass


#run a sobel filter on a 3d pointcloud
#input: pcld o3d.geometry.PointCloud
#output: pcld o3d.geometry.PointCloud with colors sobelized
def sobel_filter_3d_pointcloud(pcld):
	pass


#convert rgb to hsv
def rgb2hsv(rgb):
	assert(rgb.shape[-1] == 3)
	hsv = rgb2hsv(rgb)
	return hsv


#runs adaptive threshold on an image, taking into account surfaces using depth image
def adaptive_threshold_3d_surface(grayscale, depth, kernel_size = 81):
	assert(grayscale.shape == depth.shape)

	#print(grayscale.shape)

	timestamp = datetime.now()

	thresh_image = cv2.adaptiveThreshold(grayscale, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, kernel_size, 13)
	#contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 

	#cv2.drawContours(grayscale, contours, -1, (255, 255, 255), 3) 

	cv2.imshow('test_contours', grayscale)

	return thresh_image

#takes in a 1 channel image and outputs binary thresholded image
def hard_threshold_image(image, thresh, val=1.):
	assert(len(image.shape) == 2)
	_, thresh_image = cv2.threshold(image,thresh,val,cv2.THRESH_BINARY)
	return thresh_image


def main():
	test_color = cv2.imread(r'C:\Users\adam2\Desktop\votenet-docker\votenet\6.png')
	test_depth = np.array(cv2.imread(r'C:\Users\adam2\Desktop\votenet-docker\votenet\6d.png', cv2.IMREAD_UNCHANGED))
	test_gray = cv2.cvtColor(test_color, cv2.COLOR_BGR2GRAY)
	adaptive_threshold_3d_surface(test_gray, test_depth)
	#sobel_filter_2d(test_image)

	#mesh = o3d.io.read_triangle_mesh(r'C:\Users\adam2\Desktop\votenet-docker\votenet\object_tracking\test.ply')
	#sobel_filter_3d_mesh(mesh)

	#image = rgb2hsv(test_image)
	#cv2.imshow('test', image)
	#cv2.waitKey(0)


if __name__ == "__main__":
	main()