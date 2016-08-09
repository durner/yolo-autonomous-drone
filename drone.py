import libardrone.libardrone as libardrone
import time
from threading import Thread
import cv2


def main():
    drone = libardrone.ARDrone(True)
    drone.reset()

    try:
        t1 = Thread(target = getKeyInput, args = (drone,))
        t2 = Thread(target = getVideoImage, args = (drone,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except:
        print "Error: unable to start thread"

    print("Shutting down...")
    cv2.destroyAllWindows()
    drone.land()
    time.sleep(0.1)
    drone.halt()
    print("Ok.")


def getKeyInput(drone):
    global stop
    global key
    while not stop: # while 'bedingung true'
        time.sleep(0.1)

        if key == "t": # if 'bedingung true'
            drone.takeoff()
        elif key == " ":
            drone.land()
        elif key == "0":
            drone.hover()
        elif key == "w":
            drone.move_forward()
        elif key == "s":
            drone.move_backward()
        elif key == "a":
            drone.move_left()
        elif key == "d":
            drone.move_right()
        elif key == "q":
            drone.turn_left()
        elif key == "e":
            drone.turn_right()
        elif key == "8":
            drone.move_up()
        elif key == "2":
            drone.move_down()
        elif key == "c":
            stop = True
        else:
            drone.hover()

        if key != " ":
            key = ""


def getVideoImage(drone):
    global key
    global stop
    while not stop:
        try:
            # print pygame.image
            pixelarray = drone.get_image()
            pixelarray = cv2.cvtColor(pixelarray, cv2.COLOR_BGR2RGB)
            if pixelarray != None:
                cv2.imshow('frame', pixelarray)
                key = chr(cv2.waitKey(100))
        except:
            pass


if __name__ == '__main__':
    key = None
    stop = False
    main()
