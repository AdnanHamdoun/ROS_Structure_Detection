"""my_controller controller."""

from controller import Robot, Camera, Motor
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import time

TIME_STEP = 32
DRIVE_DURATION = 100  # seconds

robot = Robot()
camera = robot.getDevice('Camera')
camera.enable(500)

left_motor = robot.getDevice('left wheel')
right_motor = robot.getDevice('right wheel')

left_motor.setPosition(float('inf'))  # Set to velocity control mode
right_motor.setPosition(float('inf'))

velocity = 2.0

rospy.init_node('webots_controller')
image_pub = rospy.Publisher('/Camera/image_raw', Image, queue_size=1)
bridge = CvBridge()

start_time = robot.getTime()

while robot.step(TIME_STEP) != -1:
    # Get the camera image
    width = camera.getWidth()
    height = camera.getHeight()
    image = camera.getImage()

    # webots to opencv format
    image_array = np.frombuffer(image, np.uint8).reshape((height, width, 4))
    image_array = image_array[:, :, :3]

    # openCV image to ROS
    image_message = bridge.cv2_to_imgmsg(image_array, encoding='bgr8')
    image_pub.publish(image_message)

    # drive straight for 100 seconds
    elapsed_time = robot.getTime() - start_time
    if elapsed_time < DRIVE_DURATION:
        left_motor.setVelocity(velocity)
        right_motor.setVelocity(velocity)
    else:
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)
        print('Stopping')
        break