

YOLO Autonomous Drone - Deep Learning Person Detection
===================

The YOLO Drone localizes and follows people with the help of the YOLO Deep Network. Since in many scenes more than one person might be in the picture we added special person features to detect the right person. An convenient solution was the colour of the person's shirt. Hence, we require the "operator" of the drone to wear a yellow shirt.

## Requirements
To run this project Keras and Theano are needed for the deeplearning part. Furthermore a working libardrone must be installed. For shirt detection opencv must be installed on the system.

> **Requirements list (+ all dependencies!) (python2.7):**
> - keras (http://www.keras.io)
> - theano (http://deeplearning.net/software/theano/)
> - libardrone (https://github.com/venthur/python-ardrone)
> - opencv (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)

## YOLO Network
For the YOLO network we tweaked the original implementation of https://github.com/sunshineatnoon/Darknet.keras
To run the Network with pretrained weights we suggest to use http://pjreddie.com/media/files/yolo-tiny.weights

## Run the project
If you have all requirements as defined above you can simple run the project by entering:
```
$ python drone.py
```
Which contains the main file of the drone. Please make sure that you have an active connection to the drone via wifi.

## Switching between interfaces
If you want to switch between autonomous and manual flight you can simple change the main definition of drone.py by flipping the manual argument
```
def main():
    drone = YOLODrone(manual=False)
    drone.start()
```

## Autonomous Interface

![](master/pictures/detection_1.png?raw=true "Detection 1")![](master/pictures/detection_2.png?raw=true "Detection 2")

As already described the drone looks for persons. The interface marks persons / group of persons with read boxes. Additionally, a yellow t-shirt determines the real operator of the drone which is also highlighted in the interface. If more than one person wears a yellow shirt the drone chooses the person with the biggest area of yellow.

## Manual Interface
If you don't press any key the drone will hover at its position. Use following keys to control the drone.

Key     | Function
------- | ------- 
t       | takeoff
(space) | land
w       | move forward
s       | move backward
d       | move right
a       | move left
8       | move up
2       | move down
e       | turn right
q       | turn left
c       | stop flight

## Contributers
 - [Dominik Durner](https://github.com/durner)
 - [Christopher Helm](https://github.com/chrishelm)

## Upstream Repository
The current master of this project can be found at https://github.com/durner/yolo-autonomous-drone

## Files
- drone.py : Main file of the project. Includes the manual interface, the glue code to the autonomous interface between YOLO Network and Actuators. All multithreading and OpenCV pre-processing is handled.
- PID.py : simple PID controller interface to easily control the movements of the drone (incl. smoothing of the movements).
- YOLO.py : Set up of the YOLO Deep network in python. The subfolder utils include further needed files for the YOLO net.
- actuators.py : With the help of the localized operator the actuators calculate how the drone needs to move to center the operator and follow him.
