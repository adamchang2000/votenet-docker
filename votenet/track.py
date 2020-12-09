import open3d as o3d
from run import *
import os, argparse, cv2
from datetime import datetime

import open3d as o3d

from utils.color_util import *
from object_tracking.obj_to_pointcloud_util import convert_file_to_model
from object_tracking.obj_to_pointcloud_util import eulerAnglesToRotationMatrix

from utils.camera import D435i_camera, Azure_Kinect_camera

#sample this many points
NUM_POINTS_NETWORK = 100000


def perform_icp(box_points, pcld_input, model_pcld):

    box_points = box_points[0]

    max_correspondence_distance = 0.1

    init_transform = np.zeros((4, 4)).astype(np.float64)
    init_transform[0][3] = box_points[0][0]
    init_transform[1][3] = box_points[0][1]
    init_transform[2][3] = box_points[0][2]

    init_transform[:3,:3] = np.array(eulerAnglesToRotationMatrix(box_points[1]))
    init_transform[3][3] = 1.

    print(init_transform)
    init_transform = np.linalg.inv(init_transform).astype(np.float64)
    print('inverted')
    print(init_transform)

    #pcld_input.estimate_normals()

    #icp_result = o3d.registration.registration_colored_icp(pcld_input, model_pcld, max_correspondence_distance=max_correspondence_distance, init=init_transform)

    pcld_input.transform(init_transform)

    o3d.io.write_point_cloud('pcld.ply', pcld_input)
    o3d.io.write_point_cloud('target.ply', model_pcld)


    

def main():
    parser = argparse.ArgumentParser(description='Track an object')
    parser.add_argument('--model_path', action="store", default='test.py', help="Path of model to track")
    parser.add_argument('--network_path', action="store", default='test.py', help="Path of network state dict")
    parser.add_argument('--camera', action="store", default='d435i', help="Which camera to use, [d435i, azure_kinect]")
    parser.add_argument('--run_net', action="store_true", default=False, help="Run network or don't, default dont")
    parser.add_argument('--save_data', action="store_true", default=False, help="Save data, pointclouds and images")
    args = parser.parse_args()

    if args.run_net:
        if not args.model_path or not os.path.exists(args.model_path):
            print('model path incorrect')
            exit()

        if not args.network_path or not os.path.exists(args.network_path):
            print('network path incorrect')
            exit()

        #load model
        model_pcld = convert_file_to_model(args.model_path, scale=0.001)

        #get the network started
        setup(args.network_path)
    else:
        print('code running without running network')
    

    if args.camera == 'd435i':
        camera = D435i_camera()
    elif args.camera == 'azure_kinect':
        camera = Azure_Kinect_camera()
    else:
        print('need to select a valid camera')
        exit(1)

    
    idx = 0
    scale_output_pcld = 1.0

    output_dir = 'output/'

    clipping_distance = 3.0

    camera.start()

    # Streaming loop
    try:

        timestamp = 0

        while True:
            timestamp = datetime.now()
            print('start time ', timestamp)
        

            success, color_image, depth_image, pcld = camera.get_frame()
            if not success:
                continue

            depth_image_flatten = depth_image.flatten()

            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            #gray_image = adaptive_threshold_3d_surface(gray_image, depth_image)
            #gray_image = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
            gray_image = adaptive_threshold_3d_surface(gray_image, depth_image)

            #rescale from 0-255 to 0-1
            gray_image_flatten = gray_image.flatten() / 255.
            gray_image_flatten = gray_image_flatten.reshape((gray_image_flatten.shape[0], 1))

            #print(pcld.shape)
            #print(gray_image_flatten.shape)

            pcld *= scale_output_pcld

            #xyzrgb
            pcld_input = np.hstack((pcld, gray_image_flatten))
            #get rid of bad depth measurements
            pcld_input = pcld_input[abs(depth_image_flatten - clipping_distance / 2) < clipping_distance / 2].astype(np.float32)
            print('non-zero depth points ', pcld_input.shape)


            if pcld_input.shape[0] > NUM_POINTS_NETWORK:

                index = np.random.choice(pcld_input.shape[0], NUM_POINTS_NETWORK, replace=False)
                pcld_input = pcld_input[index]

                print('this should be [0, 1]:')
                print(np.unique(pcld_input.T[3]))

                # save the image frame

                save_pcld = o3d.geometry.PointCloud()
                save_pcld.points = o3d.utility.Vector3dVector(pcld_input[:,:3])

                rgb_colors = np.zeros((pcld_input.shape[0], 3))
                rgb_colors[:,0] = pcld_input[:,3]
                rgb_colors[:,1] = pcld_input[:,3]
                rgb_colors[:,2] = pcld_input[:,3]

                save_pcld.colors = o3d.utility.Vector3dVector(rgb_colors)

                if args.run_net:
                    #run network on image
                    max_conf, estimation = run_network(pcld_input)
                    required_conf = 0.95

                    if max_conf > required_conf:
                        perform_icp(estimation, save_pcld, model_pcld)


            # Render images
            cv2.namedWindow('depth', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('color', cv2.WINDOW_AUTOSIZE)
            cv2.namedWindow('gray', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('depth', depth_image)
            cv2.imshow('color', color_image)
            cv2.imshow('gray', gray_image)
                

            if args.save_data:

                #saving stuff
                o3d.io.write_point_cloud(os.path.join(output_dir, str(idx) + '.ply'), save_pcld, write_ascii=True)

                
                cv2.imwrite(os.path.join(output_dir, str(idx)+'.png'), color_image)
                #cv2.imwrite(os.path.join(output_dir, str(idx)+'d.png'), depth_image)
            key = cv2.waitKey(1)

            print('elapsed time for frame: ', datetime.now() - timestamp)

            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

            idx += 1
    except:
        raise 
    finally:
        camera.stop()


if __name__ == "__main__":
    main()