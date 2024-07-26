Welcome to the ROS Structure Detection Repository
This repository provides a simple bottle segmentation model using U-Net, implemented into ROS-Webots.

In the Main.ipynb file (jupyter notebook) you can find the extraction of the required bottle subset of the data from PartImageNet dataset, and the creation of the U-Net model with visualizations.

Webcam_Test.py provides a script to run the model locally from your webcam. Make sure the model (unet_bottle_segmentation.h5) is in the same folder as the script, otherwise adjust accordingly.

world_file folder contains the Webots world configuration. This folder is required to run the ROS package object_segmentation. Make sure to reconfigure the pathing in the package to your system.
  