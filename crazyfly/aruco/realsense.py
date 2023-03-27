import pyrealsense2 as rs
import numpy as np
import cv2
import cv2.aruco as aruco
import time
import sys
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
rate = 60
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, rate)
# config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 400)


config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, rate)

# Start streaming
pipeline.start(config)
start_time = time.time()
mtx = np.array([424.91497802734375, 0.0, 424.3425598144531, 0.0, 424.91497802734375, 238.58155822753906, 0.0, 0.0, 1.]).reshape(3,3)
distCoeffs = np.array([0.00, -0.00, -0.00, 0.00])
h, w = 480, 640
mtx, roi = cv2.getOptimalNewCameraMatrix(
         mtx, distCoeffs, (h, w), 0.0, (h, w)
    )
try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        # depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        color_profile = color_frame.get_profile()
        cvsprofile = rs.video_stream_profile(color_profile)
        color_intrin = cvsprofile.get_intrinsics()
        print(color_intrin)
        # print(frames.as_motion_frame().get_motion_data())
        # if not depth_frame or not color_frame:
        #     continue

        # Convert images to numpy arrays
        # depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # print(color_image.shape)
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        # gray = cv2.equalizeHist(gray)
        # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        # img_dst_clahe = clahe.apply(gray)

        parameters = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(aruco_dict, parameters)
        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(gray)
        if markerIds is not None:
            if [0] in markerIds:
                print(np.where(markerIds==[0])[0])
                markerCorners = markerCorners[np.where(markerIds==[0])[0][0]:np.where(markerIds==[0])[0][0]+1]
                markerIds = np.array([[0]])
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.035, mtx, distCoeffs)
                # sys.stdout.write("\r translation is :{0}".format(tvec))
                # sys.stdout.flush()
                print(tvec.shape)
                # from camera coeficcients
                (rvec - tvec).any()  # get rid of that nasty numpy value array error

                #        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #绘制轴
                #        aruco.drawDetectedMarkers(frame, corners) #在标记周围画一个正方形

                for i in range(rvec.shape[0]):
                    frame = cv2.drawFrameAxes(color_image, mtx, distCoeffs, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(frame, markerCorners)
                aruco.drawDetectedMarkers(color_image, markerCorners, markerIds)
                pos_x,pos_y,pos_z = tvec[0][0]
                text_pos = "Position" + str((round(pos_x, 2), round(pos_y, 2), round(pos_z, 2)))
                cv2.putText(color_image, text_pos, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        else:
            print("No markers found")
        # aruco.drawDetectedMarkers(color_image, rejectedCandidates, borderColor=(100, 0, 240))

        cv2.imshow('frame', color_image)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_HOT)

        # depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        # if depth_colormap_dim != color_colormap_dim:
        #     resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        #     images = np.hstack((resized_color_image, depth_colormap))
        # else:
        #     images = np.hstack((color_image, depth_colormap))

        # Show images
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RealSense', images)
        cv2.waitKey(1)
        # print("FPS: ", 1.0 / (time.time() - start_time))
        # start_time = time.time()

finally:

    # Stop streaming
    pipeline.stop()