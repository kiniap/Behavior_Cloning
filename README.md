# Behavior_Cloning

This project uses a simulator to collect data of good driving behavior to train a convolution neural network model to drive a car around a track without leaving the road.

The write_report explains the various steps taken to collect the data and to build and train the network.

## Packages needed for Python 3
* csv
* openCV2
* random
* sklearn
* Numpy
* Pandas
* Keras
* Tensorflow
* os
* socketIO
* eventlet
* PIL
* h5py

The network was trained on using AWS GPUs. It may take a long time to run on a machine without GPU support.

## Files included

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results
* video_track1.mp4 is a low resolution video of the car driving a lap on track 1

## Drive the car autonomously around the track
```sh
python drive.py model.h5
```
To visualize the run a simulator is needed, which is not included in this folder.

## Results
The project successfully trains a convolution neural network model to drive a car around a track. A low resolution video of this run is included.
