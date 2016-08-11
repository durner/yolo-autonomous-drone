import os
import libardrone.libardrone as libardrone
import time
from threading import Thread, Lock, Condition
import cv2
import numpy
import keras
import copy
from PID import PID
from YOLO import SimpleNet, convert_yolo_detections, do_nms_sort, draw_detections
from actuators import Actuator, VerticalActuator
from utils.TinyYoloNet import ReadTinyYOLONetWeights
from utils.crop import crop


class YOLODrone(object):
    def __init__(self, manuel=True):
        self.key = None
        self.stop = False
        self.mutex = None
        self.manuel = manuel
        self.PID = None
        self.boxes = None
        self.condition = Condition()
        self.update = False

        self.labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
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

        self.model = SimpleNet(yoloNet)
        sgd = keras.optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy')

    def start(self):
        self.drone = libardrone.ARDrone(True)
        self.drone.reset()

        if self.manuel:
            try:
                self.mutex = Lock()
                t1 = Thread(target=self.getKeyInput, args=())
                t2 = Thread(target=self.getVideoStream, args=())
                t3 = Thread(target=self.getBoundingBoxes, args=())
                t1.start()
                t2.start()
                t3.start()
                t1.join()
                t2.join()
                t3.join()
            except:
                print "Error: unable to start thread"
        else:
            try:
                self.mutex = Lock()
                t1 = Thread(target=self.autonomousFlight, args=(640, 360, 98, 0.1, self.labels,))
                t2 = Thread(target=self.getVideoStream, args=())
                t3 = Thread(target=self.getBoundingBoxes, args=())
                t1.start()
                t2.start()
                t3.start()
                t1.join()
                t2.join()
                t3.join()
            except:
                print "Error: unable to start thread"
            PID()


        print("Shutting down...")
        cv2.destroyAllWindows()
        self.drone.land()
        time.sleep(0.1)
        self.drone.halt()
        print("Ok.")


    def getKeyInput(self):
        while not self.stop:  # while 'bedingung true'
            time.sleep(0.1)


            if self.key == "t":  # if 'bedingung true'
                self.drone.takeoff()
            elif self.key == " ":
                self.drone.land()
            elif self.key == "0":
                self.drone.hover()
            elif self.key == "w":
                self.drone.move_forward()
            elif self.key == "s":
                self.drone.move_backward()
            elif self.key == "a":
                self.drone.move_left()
            elif self.key == "d":
                self.drone.move_right()
            elif self.key == "q":
                self.drone.turn_left()
            elif self.key == "e":
                self.drone.turn_right()
            elif self.key == "8":
                self.drone.move_up()
            elif self.key == "2":
                self.drone.move_down()
            elif self.key == "c":
                self.stop = True
            else:
                self.drone.hover()

            if self.key != " ":
                self.key = ""

    def getVideoStream(self):
        while not self.stop:
            filename = "frame"
            filename += ".jpg"
            self.mutex.acquire()
            img = cv2.imread("results/" + filename)
            self.mutex.release()
            if img != None:
                cv2.imshow('frame', img)
                l = cv2.waitKey(150)
                if l < 0:
                    self.key = ""
                else:
                    self.key = chr(l)

    def getBoundingBoxes(self):
        newest = time.time()
        while not self.stop:
            try:
                pixelarray = self.drone.get_image()
                pixelarray = cv2.cvtColor(pixelarray, cv2.COLOR_BGR2RGB)
                if pixelarray != None and newest < (time.time() - 0.15):
                    newest = time.time()
                    filename = "frame"
                    filename += ".jpg"
                    cv2.imwrite("frames/" + filename, pixelarray)

                    image = crop("frames/" + filename, resize_width=512, resize_height=512, new_width=448,
                                 new_height=448)
                    image = numpy.expand_dims(image, axis=0)
                    #

                    out = self.model.predict(image)

                    predictions = out[0]
                    boxes = convert_yolo_detections(predictions)

                    # self.condition.acquire()
                    self.boxes = do_nms_sort(boxes, 98)
                    self.update = True

                    # self.condition.notify()
                    # self.condition.release()


                    #
                    # mutex.acquire()
                    draw_detections("frames/" + filename, 98, 0.1, boxes, 20, self.labels, filename)
                    # mutex.release()
                    # time.sleep(0.2)
            except:
                pass

    def autonomousFlight(self, img_width, img_height, num, thresh, labels):
        actuator = Actuator(self.drone, img_width, img_width * 0.3)
        verticalActuator = VerticalActuator(self.drone)

        print self.drone.navdata
        while not self.stop:
            if self.update == True:
                self.update = False
                # self.condition.acquire()
                #self.condition.wait()
                boxes = copy.deepcopy(self.boxes)
                # self.condition.release()
                best_prob = -99999
                best_box = -1
                for i in range(num):
                    # for each box, find the class with maximum prob
                    max_class = numpy.argmax(boxes[i].probs)
                    prob = boxes[i].probs[max_class]
                    if prob > thresh and labels[max_class] == "person":
                        if best_prob < prob:
                            best_prob = prob
                            best_box = i

                if best_box < 0:
                    self.drone.at(libardrone.at_pcmd, False, 0, 0, 0, 0)
                    continue

                b = boxes[best_box]

                temp = b.w
                b.w = b.h
                b.h = temp

                left = (b.x - b.w / 2.) * img_width
                right = (b.x + b.w / 2.) * img_width

                top = (b.y - b.h / 2.) * img_height
                bot = (b.y + b.h / 2.) * img_height


                if (left < 0): left = 0;
                if (right > img_width - 1): right = img_width - 1;
                if (top < 0): top = 0;
                if (bot > img_height - 1): bot = img_height - 1;

                width = right - left
                print width
                print right - width/2
                height = bot - top

                actuator.step(right - width/2, width)


def main():
    drone = YOLODrone(manuel=False)
    drone.start()


if __name__ == '__main__':
    main()
