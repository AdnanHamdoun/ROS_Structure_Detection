# Run this code to test with your webcam, do not use jupyterlab for this code
# To close the window cleanly press q
# Make sure the model is in the same directiory, otherwise modify the path
# Model performs best when the bottle is placed in a flat background


import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
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


# Load the saved model
model = load_model('unet_bottle_segmentation.h5',
                   custom_objects={'combined_loss': combined_loss, 'iou': iou, 'iou_thresholded': iou_thresholded})

def preprocess_frame(frame):
    # match frames to expected size
    resized_frame = cv2.resize(frame, (256, 256))
    # normalize
    normalized_frame = resized_frame / 255.0
    # Expand dimensions
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame


# Open a connection to the webcam
cap = cv2.VideoCapture(0)


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture image.")
        break

    input_frame = preprocess_frame(frame)
    prediction = model.predict(input_frame)
    mask = (prediction > 0.5).astype(np.uint8).squeeze()
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    # Overlay the mask on the original frame
    overlay = frame.copy()
    overlay[mask_resized == 1] = (0, 255, 0)  # Green color for the mask

    # Blend the original frame and the overlay
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

    # Display the resulting frame
    cv2.imshow('Bottle Segmentation', blended)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
