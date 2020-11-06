import open3d as o3d
from run import *
import os, argparse, cv2
from datetime import datetime

import pyrealsense2 as rs
import open3d as o3d


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
    setup(args.network_path)

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
    clipping_distance_in_meters = 3
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    idx = 0

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


            color_image_flatten = color_image.reshape((color_image.shape[0] * color_image.shape[1], 3)) / 255.
            #print(color_image_flatten[150000])

            depth_image_flatten = depth_image.flatten()

            #xyzrgb
            pcld_input = np.hstack((pcld, color_image_flatten))
            #get rid of bad depth measurements
            pcld_input = pcld_input[depth_image_flatten > 0].astype(np.float32)
            print('non-zero depth points ', pcld_input.shape)

            save_pcld = o3d.geometry.PointCloud()
            save_pcld.points = o3d.utility.Vector3dVector(pcld_input[:,:3])

            rgb_colors = np.zeros((pcld_input.shape[0], 3))
            rgb_colors[:,0] = pcld_input[:,5]
            rgb_colors[:,1] = pcld_input[:,4]
            rgb_colors[:,2] = pcld_input[:,3]

            save_pcld.colors = o3d.utility.Vector3dVector(rgb_colors)
            o3d.io.write_point_cloud(str(idx) + '.ply', save_pcld, write_ascii=True)

            #run network on image
            # if pcld_input.shape[0] > 30000:
            #     run_network(pcld_input)

            # Render images
            cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth', depth_image)
            cv2.imshow('color', color_image)
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