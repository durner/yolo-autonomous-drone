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

class GoogleNet:
    layers = []
    layer_number = 29
    def __init__(self):
        self.layers.append(layer(0,0,0,0,0,"CROP"))
        self.layers.append(convolutional_layer(7,3,64,224,224))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(3,64,192,56,56))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(1,192,128,28,28))
        self.layers.append(convolutional_layer(3,128,256,28,28))
        self.layers.append(convolutional_layer(1,256,256,28,28))
        self.layers.append(convolutional_layer(3,256,512,28,28))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(1,512,256,14,14))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(convolutional_layer(1,512,256,14,14))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(convolutional_layer(1,512,256,14,14))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(convolutional_layer(1,512,256,14,14))
        self.layers.append(convolutional_layer(3,256,512,14,14))
        self.layers.append(convolutional_layer(1,512,512,14,14))
        self.layers.append(convolutional_layer(3,512,1024,14,14))
        self.layers.append(layer(0,0,0,0,0,"MAXPOOL"))
        self.layers.append(convolutional_layer(1,1024,512,7,7))
        self.layers.append(convolutional_layer(3,512,1024,7,7))
        self.layers.append(convolutional_layer(1,1024,512,7,7))
        self.layers.append(convolutional_layer(3,512,1024,7,7))
        self.layers.append(layer(0,0,0,0,0,"AVGPOOL"))
        self.layers.append(connected_layer(0,0,0,0,0,1024,1000))
        self.layers.append(layer(0,0,0,0,0,"SOFTMAX"))
        self.layers.append(layer(0,0,0,0,0,"COST"))

def ReadGoogleNetWeights(weight_path):
    googleNet = GoogleNet()
    type_string = "(3)float32,i4,"
    for i in range(googleNet.layer_number):
        l = googleNet.layers[i]
        if(l.type == "CONVOLUTIONAL"):
            bias_number = l.n
            weight_number = l.n*l.c*l.size*l.size
            type_string = type_string +"("+ str(bias_number) + ")float32,(" + str(weight_number) + ")float32,"
        elif(l.type == "CONNECTED"):
             bias_number = l.output_size
             weight_number = l.output_size * l.input_size
             type_string = type_string + "("+ str(bias_number) + ")float32,("+ str(weight_number)+")float32"
    #dt = np.dtype((+str(64)+")float32"))
    #type_string = type_string + ",i1"
    dt = np.dtype(type_string)
    testArray = np.fromfile(weight_path,dtype=dt)
    #print len(testArray[0])
    #write the weights read from file to GoogleNet biases and weights

    count = 2
    for i in range(0,googleNet.layer_number):
        l = googleNet.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            l.biases = np.asarray(testArray[0][count])
            count = count + 1
            l.weights = np.asarray(testArray[0][count])
            count = count+1
            googleNet.layers[i] = l
            #print i,count

    #write back to file and see if it is the same
    '''
    write_fp = open('reconstruct.weights','w')
    write_fp.write((np.asarray(testArray[0][0])).tobytes())
    write_fp.write((np.asarray(testArray[0][1])).tobytes())
    for i in range(0,googleNet.layer_number):
        l = googleNet.layers[i]
        if(l.type == "CONVOLUTIONAL" or l.type == "CONNECTED"):
            write_fp.write(l.biases.tobytes())
            write_fp.write(l.weights.tobytes())

    write_fp.close()
    '''
    return googleNet
