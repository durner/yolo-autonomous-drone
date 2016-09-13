YOLO Autonomous Drone - Deep Learning Person Detection
===================

The YOLO Drone localizes and follows people with the help of the YOLO Deep Network. Often, more than just one person might be in the picture of the drone’s camera so a standard deep learning people/body recognition cannot deliver sufficient results. This is why we chose the color of the shirt worn by the respective person to be a second criterion. Hence, we require the "operator" of the drone to wear a shirt with a distinct yellow color. This turns out to be a suitable solution to the aforementioned problem. 

## Requirements
To run this project Keras and Theano are needed for the deeplearning part. Furthermore, a working libardrone must be installed. For shirt detection opencv must be installed on the system.

> **Requirements list (+ all dependencies!) (python2.7):**
> - keras (http://www.keras.io)
> - theano (http://deeplearning.net/software/theano/)
> - libardrone (https://github.com/venthur/python-ardrone)
> - opencv (http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)

## YOLO Network
For the YOLO network we tweaked the original implementation of https://github.com/sunshineatnoon/Darknet.keras. To run the Network with pretrained weights we suggest to use http://pjreddie.com/media/files/yolo-tiny.weights.

## Run the project
If you have all requirements as defined above you can simple run the project by entering:
```
$ python drone.py
```
This contains the main file of the drone. Please make sure that you have an active connection to the drone via wifi.

## Switching between interfaces
If you want to switch between autonomous and manual flight you can simply change the main definition of drone.py by flipping the manual argument
```
def main():
    drone = YOLODrone(manual=False)
    drone.start()
```

## Autonomous Interface

![Detection 1](pictures/detection_1.png?raw=true "Detection 1") ![Detection 2](pictures/detection_2.png?raw=true "Detection 2")

As already described, the drone is looking for persons. The interface marks persons / groups of persons with red boxes. Additionally, a yellow t-shirt determines the real operator of the drone which is also highlighted in the interface. If more than one person wears a yellow shirt in the picture, the drone chooses the red box (person) that has the highest amount of yellow in them and continues to follow this particular person.

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
- actuators.py : With the help of the localized operator the actuators calculate how the drone needs to move to center the operator and follow him. Uses PID controllers for calculating the movements.
