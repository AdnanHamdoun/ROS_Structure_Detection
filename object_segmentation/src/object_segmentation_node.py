#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import tensorflow as tf
import time
import csv
from keras.models import load_model
from keras_unet.metrics import iou, iou_thresholded


def dice_coefficient(y_true, y_pred):
    smooth = 1e-6
    y_true_f = tf.keras.backend.flatten(tf.cast(y_true, tf.float32))
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice_loss = 1 - dice_coefficient(y_true, y_pred)
    return bce + dice_loss


model = load_model('/home/adnan/catkin_ws/src/object_segmentation/models/unet_bottle_segmentation.h5',
                   custom_objects={'combined_loss': combined_loss, 'iou': iou, 'iou_thresholded': iou_thresholded})

# Initialize CvBridge
bridge = CvBridge()


def preprocess_image(cv_image):
    # Resize input image
    input_size = (256, 256)
    resized_image = cv2.resize(cv_image, input_size)
    # Normalize
    normalized_image = resized_image / 255.0
    # Expand dimensions
    input_image = np.expand_dims(normalized_image, axis=0)
    return input_image


def postprocess_output(output):
    # Grayscale
    output = output.squeeze()
    # Binary mask
    _, segmented_image = cv2.threshold(output, 0.5, 1, cv2.THRESH_BINARY)
    return segmented_image


# Initialize CSV writing
csv_filename = '/home/adnan/Desktop/timing_data.csv'
csv_file = open(csv_filename, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Time', 'ROS to OpenCV', 'Preprocessing', 'Prediction', 'Postprocessing'])

# Stop processing at 100s
start_time = time.time()


def image_callback(msg):
    global start_time

    elapsed_time = time.time() - start_time
    if elapsed_time > 100:
        rospy.signal_shutdown("100_seconds")
        csv_file.close()
        return

    # ROS to OpenCV
    start_ros_to_opencv = time.time()
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")
    ros_to_opencv_time = time.time() - start_ros_to_opencv

    # Preprocess
    start_preprocessing = time.time()
    input_image = preprocess_image(cv_image)
    preprocessing_time = time.time() - start_preprocessing

    # Predictions
    start_prediction = time.time()
    output = model.predict(input_image)
    prediction_time = time.time() - start_prediction

    # Postprocess
    start_postprocessing = time.time()
    segmented_image = postprocess_output(output)
    postprocessing_time = time.time() - start_postprocessing

    # Publish image
    segmented_image_msg = bridge.cv2_to_imgmsg((segmented_image * 255).astype(np.uint8), 'mono8')
    segmented_image_pub.publish(segmented_image_msg)

    # Display mask
    cv_image_resized = cv2.resize(cv_image, (640, 480))  # Resize for display
    cv2.imshow("Segmented Image", segmented_image)
    cv2.waitKey(1)

    # Write timing data to CSV
    csv_writer.writerow([elapsed_time, ros_to_opencv_time, preprocessing_time, prediction_time, postprocessing_time])
    print(f"Data written at {elapsed_time:.2f} seconds")


def main():
    global segmented_image_pub
    # Initialize node
    rospy.init_node('object_segmentation_node')
    # Subscribe to the camera
    rospy.Subscriber('/Camera/image_raw', Image, image_callback)
    # Publish the mask images
    segmented_image_pub = rospy.Publisher('/segmented_image', Image, queue_size=10)
    rospy.spin()


if __name__ == '__main__':
    main()
