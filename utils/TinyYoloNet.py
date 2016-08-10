import numpy as np
from enum import Enum
import os

class layer:
    def __init__(self,size,c,n,h,w,type):
        self.size = size
        self.c = c
        self.n = n
        self.h = h
        self.w = w
        self.type = type

class convolutional_layer(layer):
    def __init__(self,size,c,n,h,w):
        layer.__init__(self,size,c,n,h,w,"CONVOLUTIONAL")
        self.biases = np.zeros(n)
        self.weights = np.zeros((size*size,c,n))

class connected_layer(layer):
    def __init__(self,size,c,n,h,w,input_size,output_size):
        layer.__init__(self,size,c,n,h,w,"CONNECTED")
        self.output_size = output_size
        self.input_size = input_size
        self.biases = np.zeros(output_size)
        self.weights = np.zeros((output_size*input_size))

class Tiny_YOLO:
    layers = []
    layer_number = 22
    def __init__(self):
        self.layers.append(layer(0,0,0,0,0,"CROP"))
        self.layers.append(convolutional_layer(3,3,16,448,448))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,16,32,224,224))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,32,64,112,112))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,64,128,56,56))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,128,256,28,28))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,512,1024,7,7))
        self.layers.append(convolutional_layer(3,1024,1024,7,7))
        self.layers.append(convolutional_layer(3,1024,1024,7,7))
        self.layers.append(layer(0,0,0,0,0,"FLATTEN"))
        self.layers.append(connected_layer(0,0,0,0,0,50176,256))
        self.layers.append(connected_layer(0,0,0,0,0,256,4096))
        self.layers.append(layer(0,0,0,0,0,"DROPOUT"))
        self.layers.append(layer(0,0,0,0,0,"LEAKY"))
        self.layers.append(connected_layer(0,0,0,0,0,4096,1470))

def ReadTinyYOLONetWeights(weight_path):
    YOLO = Tiny_YOLO()
    type_string = "(3)float32,i4,"
    for i in range(YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            bias_number = l.n
            weight_number = l.n*l.c*l.size*l.size
            type_string = type_string +"("+ str(bias_number) + ")float32,(" + str(weight_number) + ")float32,"
        elif(l.type == "CONNECTED"):
             bias_number = l.output_size
             weight_number = l.output_size * l.input_size
             type_string = type_string + "("+ str(bias_number) + ")float32,("+ str(weight_number)+")float32"
             if(i != YOLO.layer_number-1):
                 type_string = type_string + ","
    #dt = np.dtype((+str(64)+")float32"))
    #type_string = type_string + ",i1"
    dt = np.dtype(type_string)
    testArray = np.fromfile(weight_path,dtype=dt)
    #write the weights read from file to GoogleNet biases and weights

    count = 2
    for i in range(0,YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            l.biases = np.asarray(testArray[0][count])
            count = count + 1
            l.weights = np.asarray(testArray[0][count])
            count = count + 1
            YOLO.layers[i] = l

    #write back to file and see if it is the same
    '''
    write_fp = open('reconstruct.weights','w')
    write_fp.write((np.asarray(testArray[0][0])).tobytes())
    write_fp.write((np.asarray(testArray[0][1])).tobytes())
    for i in range(0,YOLO.layer_number):
        l = YOLO.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            write_fp.write(l.biases.tobytes())
            write_fp.write(l.weights.tobytes())


    write_fp.close()
    '''
    return YOLO

if __name__ == '__main__':
    YOLO = ReadTinyYOLONetWeights('/home/xuetingli/Documents/YOLO.keras/weights/yolo-tiny.weights')
    for i in range(YOLO.layer_number):
        l = YOLO.layers[i]
        print l.type
