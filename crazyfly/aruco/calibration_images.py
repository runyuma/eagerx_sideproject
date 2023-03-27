# Use in combination with ROS camera_calibration package: http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration
# Install
# sudo apt-get install ros-melodic-cv-camera: http://wiki.ros.org/cv_camera
# http://wiki.ros.org/camera_calibration
# Run commands:
# rosparam set cv_camera/device_id <CAM_IDX> (e.g. 0, 1, or 2).
# rosrun cv_camera cv_camera_node
# rosrun camera_calibration cameracalibrator.py --size 8x5 --square 0.0295 image:=/cv_camera/image_raw camera:=/cv_camera
# Follow steps:
# http://wiki.ros.org/camera_calibration/Tutorials/MonocularCalibration
import rospy
import atexit
import rosparam
import rosgraph

DIR = "/home/bas/Desktop/logitech_calibration"
CAM_IDX = 2


def _is_roscore_running():
    return rosgraph.is_master_online()


def _launch_roscore():
    if not _is_roscore_running():

        import roslaunch

        uuid = roslaunch.rlutil.get_or_generate_uuid(options_runid=None, options_wait_for_master=False)
        roslaunch.configure_logging(uuid)
        roscore = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_files=[], is_core=True)

        try:
            roscore.start()
        except roslaunch.core.RLException:
            rospy.loginfo(
                "Roscore cannot run as another roscore/master is already running. Continuing without re-initializing the roscore."
            )
            raise  # todo: maybe not raise here?
    else:
        rospy.loginfo(
            "Roscore cannot run as another roscore/master is already running. Continuing without re-initializing the roscore."
        )
        roscore = None
    return roscore


def _initialize(*args, log_level=rospy.INFO, **kwargs):
    roscore = _launch_roscore()  # First launch roscore (if not already running)
    try:
        rospy.init_node(*args, log_level=log_level, **kwargs)
    except rospy.exceptions.ROSException as e:
        rospy.logwarn(e)
    if roscore:
        atexit.register(roscore.shutdown)
    return roscore


_initialize("camera", anonymous=True, log_level=rospy.INFO)
rosparam.upload_params("/cv_camera/device_id", 2)  # (int: default 0) capture device id.
