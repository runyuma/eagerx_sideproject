import cv2
import numpy as np
import cv2.aruco as aruco
import sys
cap = cv2.VideoCapture(0)
mtx=np.array([[561.192444  , 0.      ,   336.597168],
 [  0.       ,  550.620117, 199.013839],
 [  0.,           0.,           1.        ]])
distCoeffs = np.array([0.082585, -0.025474, -0.023368, 0.001057])
h, w = 480, 640
mtx, roi = cv2.getOptimalNewCameraMatrix(
         mtx, distCoeffs, (h, w), 0.0, (h, w)
    )
while True:
    ret,frame = cap.read()
    # frame = cv2.resize(frame, None, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)
    print(markerIds)


    if markerIds is not None:

        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(markerCorners, 0.025, mtx, distCoeffs)
        # sys.stdout.write("\r translation is :{0}".format(tvec))
        # sys.stdout.flush()
        # print(rvec, tvec)
        # from camera coeficcients
        (rvec - tvec).any()  # get rid of that nasty numpy value array error

        #        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #绘制轴
        #        aruco.drawDetectedMarkers(frame, corners) #在标记周围画一个正方形

        for i in range(rvec.shape[0]):
            frame = cv2.drawFrameAxes(frame, mtx, distCoeffs, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, markerCorners)


    aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
    cv2.imshow('frame',frame)
    c = cv2.waitKey(1)
    if c == ord('q'):
        break
cap.release()
cv2.DestroyAllWindows()



# D = [0.08258502593635345, -0.025473523940458556, -0.023367813413300662, 0.0010568145246812892, 0.0]
# K = [535.7074339672774, 0.0, 337.09867597797387, 0.0, 536.6901264634691, 209.8276723177747, 0.0, 0.0, 1.0]
# R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# P = [561.1924438476562, 0.0, 336.5971681835108, 0.0, 0.0, 550.6201171875, 199.01383939693915, 0.0, 0.0, 0.0, 1.0, 0.0]
# None
# # oST version 5.0 parameters
#
#
# [image]
#
# width
# 640
#
# height
# 480
#
# [narrow_stereo]
#
# camera matrix
# 535.707434 0.000000 337.098676
# 0.000000 536.690126 209.827672
# 0.000000 0.000000 1.000000
#
# distortion
# 0.082585 -0.025474 -0.023368 0.001057 0.000000
#
# rectification
# 1.000000 0.000000 0.000000
# 0.000000 1.000000 0.000000
# 0.000000 0.000000 1.000000
#
# projection
# 561.192444 0.000000 336.597168 0.000000
# 0.000000 550.620117 199.013839 0.000000
# 0.000000 0.000000 1.000000 0.000000
#
# ('Wrote calibration data to', '/tmp/calibrationdata.tar.gz')

