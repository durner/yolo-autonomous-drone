import os
import libardrone.libardrone as libardrone
import time
from threading import Thread, Timer
import cv2
import numpy
import keras
from YOLO import SimpleNet, convert_yolo_detections, do_nms_sort, draw_detections
from utils.TinyYoloNet import ReadTinyYOLONetWeights
from utils.crop import crop


def main(model):
    drone = libardrone.ARDrone(True)
    drone.reset()

    try:
        t1 = Thread(target = getKeyInput, args = (drone,))
        t2 = Thread(target = getVideoImage, args = (drone,model,))
        t3 = Thread(target = getVideoKeyInput, args = ())
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
        t3.join()
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

def getVideoKeyInput():
    global stop
    global key
    while not stop:
        filename = "frame"
        filename += ".jpg"
        img = cv2.imread("results/" + filename)
        if img != None:
            cv2.imshow('frame', img)
            l = cv2.waitKey(300)
            if l < 0:
                key = ""
            else:
                key = chr(l)

def getVideoImage(drone, model):
    global stop
    global labels
    newest = time.time()
    while not stop:
        try:
            pixelarray = drone.get_image()
            pixelarray = cv2.cvtColor(pixelarray, cv2.COLOR_BGR2RGB)
            img = None
            if pixelarray != None and newest < (time.time() - 0.3):
                newest = time.time()
                filename = "frame"
                filename += ".jpg"
                cv2.imwrite("frames/" + filename, pixelarray)

                image = crop("frames/" + filename, resize_width=512, resize_height=512, new_width=448, new_height=448)
                image = numpy.expand_dims(image, axis=0)
                #
                out = model.predict(image)
                predictions = out[0]
                boxes = convert_yolo_detections(predictions)
                boxes = do_nms_sort(boxes, 98)
                #
                draw_detections("frames/" + filename,98,0.2,boxes,20,labels,filename)
                # time.sleep(0.2)
        except:
            pass


if __name__ == '__main__':
    key = None
    stop = False
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    yoloNet = ReadTinyYOLONetWeights(os.path.join(os.getcwd(), 'weights/yolo-tiny.weights'))
    # reshape weights in every layer
    for i in range(yoloNet.layer_number):
        l = yoloNet.layers[i]
        if (l.type == 'CONVOLUTIONAL'):
            weight_array = l.weights
            n = weight_array.shape[0]
            weight_array = weight_array.reshape((n // (l.size * l.size), (l.size * l.size)))[:, ::-1].reshape((n,))
            weight_array = numpy.reshape(weight_array, [l.n, l.c, l.size, l.size])
            l.weights = weight_array
        if (l.type == 'CONNECTED'):
            weight_array = l.weights
            weight_array = numpy.reshape(weight_array, [l.input_size, l.output_size])
            l.weights = weight_array

    model = SimpleNet(yoloNet)
    sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    main(model)
