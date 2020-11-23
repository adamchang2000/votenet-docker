import open3d as o3d
from run import *
import os, argparse, cv2
from datetime import datetime

import pyrealsense2 as rs
import open3d as o3d

from utils.color_util import *

#sample this many points
NUM_POINTS_NETWORK = 75000

def main():
    parser = argparse.ArgumentParser(description='Track an object')
    parser.add_argument('--model_path', action="store", help="Path of model to track")
    parser.add_argument('--network_path', action="store", help="Path of network state dict")
    args = parser.parse_args()

    if not args.model_path or not os.path.exists(args.model_path):
        print('model path incorrect')
        exit()

    if not args.network_path or not os.path.exists(args.network_path):
        print('network path incorrect')
        exit()

    #get the network started
    #setup(args.network_path)

    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)
    depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 2
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    idx = 0

    output_dir = 'output/'

    # Streaming loop
    try:

        timestamp = 0

        while True:
            timestamp = datetime.now()
            print('start time ', timestamp)
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            aligned_color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not depth_frame or not aligned_color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(aligned_color_frame.get_data())

            pc = rs.pointcloud()
            pc.map_to(aligned_color_frame)
            #pointcloud = np.asanyarray(pc.calculate(depth_frame).get_data())
            #print(pointcloud.shape)
            pointcloud = pc.calculate(depth_frame)
            pcld = np.asanyarray(pointcloud.get_vertices())
            pcld = pcld.view(np.float32).reshape(pcld.shape + (-1,))

            #minus 0.5 in accordance with training data
            color_image_flatten = color_image.reshape((color_image.shape[0] * color_image.shape[1], 3)) / 255. - 0.5

            #NORMALIZE COLORS
            #color_image_flatten /= np.max(np.abs(color_image_flatten), axis=0)


            #print(np.max(np.abs(color_image_flatten), axis=0))
            #print(color_image_flatten[150000])

            depth_image_flatten = depth_image.flatten()

            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            #gray_image = adaptive_threshold_3d_surface(gray_image, depth_image)
            gray_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
            gray_image_flatten = gray_image.flatten()
            gray_image_flatten = gray_image_flatten.reshape((gray_image_flatten.shape[0], 1))

            #print(pcld.shape)
            #print(gray_image_flatten.shape)

            #xyzrgb
            pcld_input = np.hstack((pcld, gray_image_flatten))
            #get rid of bad depth measurements
            pcld_input = pcld_input[depth_image_flatten > 0].astype(np.float32)
            print('non-zero depth points ', pcld_input.shape)


            if pcld_input.shape[0] > NUM_POINTS_NETWORK:

                index = np.random.choice(pcld_input.shape[0], NUM_POINTS_NETWORK, replace=False)
                pcld_input = pcld_input[index]

                #run network on image
                #run_network(pcld_input)

            # save the image frame

            save_pcld = o3d.geometry.PointCloud()
            save_pcld.points = o3d.utility.Vector3dVector(pcld_input[:,:3])

            rgb_colors = np.zeros((pcld_input.shape[0], 3))
            rgb_colors[:,0] = pcld_input[:,3]
            rgb_colors[:,1] = pcld_input[:,3]
            rgb_colors[:,2] = pcld_input[:,3]

            save_pcld.colors = o3d.utility.Vector3dVector(rgb_colors)
            o3d.io.write_point_cloud(os.path.join(output_dir, str(idx) + '.ply'), save_pcld, write_ascii=True)

            # Render images
            cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('gray', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth', depth_image)
            cv2.imshow('color', color_image)
            cv2.imshow('gray', gray_image)
            cv2.imwrite(os.path.join(output_dir, str(idx)+'.png'), color_image)
            cv2.imwrite(os.path.join(output_dir, str(idx)+'d.png'), depth_image)
            key = cv2.waitKey(1)

            print('elapsed time for frame: ', datetime.now() - timestamp)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

            idx += 1

    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()