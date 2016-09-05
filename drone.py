import os
import libardrone.libardrone as libardrone
import time
from threading import Thread, Lock, Condition
import cv2
import numpy
import keras
from YOLO import SimpleNet, convert_yolo_detections, do_nms_sort
from actuators import Actuator
from utils.TinyYoloNet import ReadTinyYOLONetWeights


class YOLODrone(object):
    def __init__(self, manual=True):
        self.key = None
        self.stop = False
        self.mutex = None
        self.manuel = manual
        self.PID = None
        self.boxes = None
        self.condition = Condition()
        self.update = False
        self.contours = None
        self.boxes_update = False
        self.image = None

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
                t1 = Thread(target=self.autonomousFlight, args=(448, 448, 98, 0.1, self.labels,))
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

    def getVideoStream(self, img_width=448, img_height=448):
        while not self.stop:
            img = self.image
            if img != None:
                nav_data = self.drone.get_navdata()
                nav_data = nav_data[0]
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_size = 0.5

                cv2.putText(img, 'Altitude: %.0f' % nav_data['altitude'], (5, 15), font, font_size, (255, 255, 255))
                cv2.putText(img, 'Battery: %.0f%%' % nav_data['battery'], (5, 30), font, font_size, (255, 255, 255))

                cv2.drawContours(img, self.contours, -1, (0, 255, 0), 3)
                thresh = 0.2
                self.mutex.acquire()
                if self.boxes_update:
                    self.boxes_update = False
                    for b in self.boxes:
                        max_class = numpy.argmax(b.probs)
                        prob = b.probs[max_class]
                        if (prob > thresh and self.labels[max_class] == "person"):
                            left = (b.x - b.w / 2.) * img_width
                            right = (b.x + b.w / 2.) * img_width

                            top = (b.y - b.h / 2.) * img_height
                            bot = (b.y + b.h / 2.) * img_height

                            cv2.rectangle(img, (int(left), int(top)), (int(right), int(bot)), (0, 0, 255), 3)
                self.mutex.release()
                cv2.imshow('frame', img)

                l = cv2.waitKey(150)
                if l < 0:
                    self.key = ""
                else:
                    self.key = chr(l)
                    if self.key == "c":
                        self.stop = True

    def variance_of_laplacian(self, image):
        # compute the Laplacian of the image and then return the focus
        #  measure, which is simply the variance of the Laplacian
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def getBoundingBoxes(self):
        newest = time.time()
        while not self.stop:
            try:
                pixelarray = self.drone.get_image()
                pixelarray = cv2.cvtColor(pixelarray, cv2.COLOR_BGR2RGB)

                # Check for Blurry
                gray = cv2.cvtColor(pixelarray, cv2.COLOR_RGB2GRAY)
                fm = self.variance_of_laplacian(gray)
                if fm < 10:
                    continue

                if pixelarray != None:
                    # ima = pixelarray[120:540]
                    ima = cv2.resize(pixelarray, (448, 448))

                    image = cv2.cvtColor(ima, cv2.COLOR_RGB2BGR)

                    image = numpy.rollaxis(image, 2, 0)
                    image = image / 255.0
                    image = image * 2.0 - 1.0
                    image = numpy.expand_dims(image, axis=0)

                    out = self.model.predict(image)
                    predictions = out[0]
                    boxes = convert_yolo_detections(predictions)

                    self.mutex.acquire()
                    self.boxes = do_nms_sort(boxes, 98)
                    self.image = ima
                    self.update = True
                    self.mutex.release()

            except:
                pass

    def autonomousFlight(self, img_width, img_height, num, thresh, labels):
        actuator = Actuator(self.drone, img_width, img_width * 0.5)

        print self.drone.navdata
        while not self.stop:
            if self.update == True:
                self.update = False

                hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
                image = cv2.medianBlur(hsv, 3)

                # Filter by color red
                lower_red_1 = numpy.array([15, 150, 150])
                upper_red_1 = numpy.array([35, 255, 255])

                image = cv2.inRange(image, lower_red_1, upper_red_1)

                # Put on median blur to reduce noise
                image = cv2.medianBlur(image, 11)

                # Find contours and decide if hat is one of them
                contours, hierarchy = cv2.findContours(image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

                self.contours = contours

                boxes = self.boxes

                best_prob = -99999
                best_box = -1
                best_contour = None

                self.mutex.acquire()
                for i in range(num):
                    # for each box, find the class with maximum prob
                    max_class = numpy.argmax(boxes[i].probs)
                    prob = boxes[i].probs[max_class]

                    temp = boxes[i].w
                    boxes[i].w = boxes[i].h
                    boxes[i].h = temp

                    if prob > thresh and labels[max_class] == "person":

                        for contour in contours:
                            x, y, w, h = cv2.boundingRect(contour)

                            left = (boxes[i].x - boxes[i].w / 2.) * img_width
                            right = (boxes[i].x + boxes[i].w / 2.) * img_width

                            top = (boxes[i].y - boxes[i].h / 2.) * img_height
                            bot = (boxes[i].y + boxes[i].h / 2.) * img_height

                            if not (x + w < left or right < x or y + h < top or bot < y):
                               if best_prob < prob and w > 30:
                                    print "prob found"
                                    best_prob = prob
                                    best_box = i
                                    best_contour = contour

                self.boxes_update = True
                if best_box < 0:
                    # print "No Update"
                    self.mutex.release()
                    self.drone.at(libardrone.at_pcmd, False, 0, 0, 0, 0)
                    continue

                b = boxes[best_box]

                left = (b.x - b.w / 2.) * img_width
                right = (b.x + b.w / 2.) * img_width

                top = (b.y - b.h / 2.) * img_height
                bot = (b.y + b.h / 2.) * img_height


                if (left < 0): left = 0;
                if (right > img_width - 1): right = img_width - 1;
                if (top < 0): top = 0;
                if (bot > img_height - 1): bot = img_height - 1;

                width = right - left
                height = bot - top
                x, y, w, h = cv2.boundingRect(best_contour)

                actuator.step(right - width/2., width)
                self.mutex.release()


def main():
    drone = YOLODrone(manual=False)
    drone.start()


if __name__ == '__main__':
    main()
