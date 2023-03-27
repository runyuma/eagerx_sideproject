import rospy
from sensor_msgs.msg import Image
import time
from cv_bridge import CvBridge
import cv2
import numpy as np
from cv2 import aruco
class rs2_ros:

    def __init__(self):
        rospy.init_node('rs2_localize', anonymous=True)
        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.get_image_callback)
        self.rate = rospy.Rate(60) #
        self.image = None
        mtx = np.array(
            [610.408, 0.0, 328.45, 0.0, 609.199, 237.006, 0.0, 0.0,
             1.]).reshape(3, 3)
        self.distCoeffs = np.array([0.00, -0.00, -0.00, 0.00])
        h, w = 480, 640
        self.mtx, roi = cv2.getOptimalNewCameraMatrix(
                 mtx, self.distCoeffs, (h, w), 0.0, (h, w)
            )
        aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        parameters = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(aruco_dict, parameters)
        self.step = 0

    def main(self):
        while not rospy.is_shutdown():
            if self.image is not None:
                print(self.image.shape)
            self.rate.sleep()
    def get_image_callback(self, msg):
        self.start_time = time.time()
        cv_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        markerCorners, markerIds, rejectedCandidates = self.detector.detectMarkers(gray)
        if markerIds is not None:
            if [0] in markerIds:
                markerCorners = markerCorners[np.where(markerIds == [0])[0][0]:np.where(markerIds == [0])[0][0] + 1]
                markerIds = np.array([[0]])

                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.035, self.mtx, self.distCoeffs)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error


                for i in range(rvec.shape[0]):
                    frame = cv2.drawFrameAxes(cv_img, self.mtx, self.distCoeffs, rvec[i, :, :], tvec[i, :, :], 0.03)
                    aruco.drawDetectedMarkers(frame, markerCorners)
                aruco.drawDetectedMarkers(cv_img, markerCorners, markerIds)
                pos_x, pos_y, pos_z = tvec[0][0]
                pos_x= pos_x
                pos_y = -pos_y
                pos_z = -pos_z
                text_pos = "Position" + str((round(pos_x, 2), round(pos_y, 2), round(pos_z, 2)))
                cv2.putText(cv_img, text_pos, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
        else:
            print("No markers found")
        print(self.start_time - time.time())
        cv2.imshow("cv_img", cv_img)
        cv2.waitKey(1)


node = rs2_ros()
node.main()
