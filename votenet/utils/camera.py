import numpy as np
import cv2


#import rs
try:
    import pyrealsense2 as rs
except:
    print("Import pyrealsense2 failed, d453i camera won't work")

try:
    import pyk4a
    from pyk4a import Config, PyK4A
except:
    print("Import pyk4a failed, azure kinect camera won't work")


class Camera():
    def __init__(self):
        return

    #return next frame
    #success, BGR aligned to depth, depth, depth_pcld, IR
    def get_frame(self):
        return False, 0, 0, 0, 0

    def start(self):
        return

    def stop(self):
        return


class D435i_camera(Camera):

    def __init__(self):

        # Create a pipeline
        self.pipeline = rs.pipeline()

        
    def start(self):

        #Create a config and configure the pipeline to stream
        #  different resolutions of color and depth streams
        config = rs.config()
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
        config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)

        # Start streaming
        profile = self.pipeline.start(config)

        # Getting the depth sensor's depth scale (see rs-align example for explanation)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("Depth Scale is: " , self.depth_scale)
        depth_intrin = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        # Create an align object
        # rs.align allows us to perform alignment of depth frames to others frames
        # The "align_to" is the stream type to which we plan to align depth frames.
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame(self):
        # Get frameset of color and depth
        frames = self.pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = self.align.process(frames)

        # Get aligned frames
        depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        aligned_color_frame = aligned_frames.get_color_frame()
        ir_frame = np.asarray(frames.get_infrared_frame().get_data()).astype(np.uint16) * 257

        # Validate that both frames are valid
        if not depth_frame or not aligned_color_frame:
            return False, 0, 0, 0, 0

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        pc = rs.pointcloud()
        pc.map_to(aligned_color_frame)
        #pointcloud = np.asanyarray(pc.calculate(depth_frame).get_data())
        #print(pointcloud.shape)
        pointcloud = pc.calculate(depth_frame)
        pcld = np.asanyarray(pointcloud.get_vertices())
        pcld = pcld.view(np.float32).reshape(pcld.shape + (-1,))

        depth_image = depth_image.astype(np.float32) * self.depth_scale

        return True, color_image, depth_image, pcld, ir_frame

    def stop(self):
        self.pipeline.stop()


class Azure_Kinect_camera(Camera):
    def __init__(self):
        self.k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_720P,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
        )

    def start(self):

        self.k4a.start()

        # getters and setters directly get and set on device
        self.k4a.whitebalance = 3500
        assert self.k4a.whitebalance == 3500

    def get_frame(self):
        capture = self.k4a.get_capture()
        if np.any(capture.color):

            depth = capture.depth.astype(np.float32) / 1000.
            color = capture.color
            #ir = capture.transformed_ir
            ir = capture.ir
            transformed_color = capture.transformed_color
            pcld = capture.depth_point_cloud.astype(np.float32) / 1000.
            pcld = pcld.reshape((pcld.shape[0] * pcld.shape[1], pcld.shape[2]))

            return True, transformed_color, depth, pcld, ir

        else:
            return False, 0, 0, 0, 0


    def stop(self):
        self.k4a.stop()