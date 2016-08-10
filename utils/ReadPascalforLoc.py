import os
import xml.etree.ElementTree as ET
from crop import crop_detection
import numpy as np
from PIL import Image
from PIL import ImageDraw
import scipy
import random


vocPath = os.path.abspath(os.path.join(os.getcwd(),os.path.pardir,'dataset'))

class box:
    def __init__(self,x,y,h,w):
        self.x = x
        self.y = y
        self.h = h
        self.w = w

def overlap(x1,w1,x2,w2):
    l1 = x1 - w1/2;
    l2 = x2 - w2/2;
    if(l1 > l2):
        left = l1
    else:
        left = l2
    r1 = x1 + w1/2;
    r2 = x2 + w2/2;
    if(r1 < r2):
        right = r1
    else:
        right = r2
    return right - left;

def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w);
    h = overlap(a.y, a.h, b.y, b.h);
    if(w < 0 or h < 0):
         return 0;
    area = w*h;
    return area;

def box_union(a, b):
    i = box_intersection(a, b);
    u = a.w*a.h + b.w*b.h - i;
    return u;

def box_iou(a, b):
    return box_intersection(a, b)/box_union(a, b);

class image():
    """
    Args:
       side: An image is divided into side*side grids
    Each image class has two variables:
       imgPath: the path of an image on my computer
       bboxes: a side*side matrix, each element in the matrix is cell
    """
    def __init__(self,side,imgPath):
        self.imgPath = imgPath
        self.boxes = np.zeros((side,side))

    def parseXML(self,xmlPath,labels,side):
        """
        Args:
          xmlPath: The path of the xml file of this image
          labels: label names of pascal voc dataset
          side: an image is divided into side*side grid
        """
        tree = ET.parse(xmlPath)
        root = tree.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        for obj in root.iter('object'):
            class_num = labels.index(obj.find('name').text)
            bndbox = obj.find('bndbox')
            left = int(bndbox.find('xmin').text)
            top = int(bndbox.find('ymin').text)
            right = int(bndbox.find('xmax').text)
            down = int(bndbox.find('ymax').text)

            #trans the coords to 448*448
            left = left*1.0 / width * 448
            right = right*1.0 / width * 448
            top = top*1.0 / height * 448
            down = down*1.0 / height * 448

            boxa = box((left+right)/2,(top+down)/2,down-top,right-left)
            for i in range(int(left/64),int(right/64)+1):
                for j in range(int(top/64),int(down/64)+1):
                    box_left = i*64
                    box_right = (i+1)*64
                    box_top = j*64
                    box_down = (j+1)*64
                    boxb = box((box_left+box_right)/2,(box_top+box_down)/2,64,64)
                    iou = box_intersection(boxa,boxb)
                    if(iou/(64*64) > 0.25):
                        self.boxes[j][i] = 1

def prepareBatch(start,end,imageNameFile,vocPath):
    """
    Args:
      start: the number of image to start
      end: the number of image to end
      imageNameFile: the path of the file that contains image names
      vocPath: the path of pascal voc dataset
    Funs:
      generate a batch of images from start~end
    Returns:
      A list of end-start+1 image objects
    """
    imageList = []
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    file = open(imageNameFile)
    imageNames = file.readlines()
    for i in range(start,end):
        imgName = imageNames[i].strip('\n')
        imgPath = os.path.join(vocPath,'JPEGImages',imgName)+'.jpg'
        xmlPath = os.path.join(vocPath,'Annotations',imgName)+'.xml'
        img = image(side=7,imgPath=imgPath)
        img.parseXML(xmlPath,labels,7)
        imageList.append(img)

    return imageList

#Prepare training data
def generate_batch_data(vocPath,imageNameFile,batch_size,sample_number):
    """
    Args:
      vocPath: the path of pascal voc data
      imageNameFile: the path of the file of image names
      batchsize: batch size, sample_number should be divided by batchsize
    Funcs:
      A data generator generates training batch indefinitely
    """
    class_num = 20
    #Read all the data once and dispatch them out as batches to save time
    TotalimageList = prepareBatch(0,sample_number,imageNameFile,vocPath)

    while 1:
        batches = sample_number // batch_size
        for i in range(batches):
            images = []
            boxes = []
            sample_index = np.random.choice(sample_number,batch_size,replace=True)
            #sample_index = [3]
            for ind in sample_index:
                image = TotalimageList[ind]
                #print image.imgPath
                image_array = crop_detection(image.imgPath,new_width=448,new_height=448)
                #image_array = np.expand_dims(image_array,axis=0)

                images.append(image_array)
                boxes.append(np.reshape(image.boxes,-1))
            #return np.asarray(images),np.asarray(boxes)
            yield np.asarray(images),np.asarray(boxes)

if __name__ == '__main__':
    img,boxes = generate_batch_data('/home/media/Documents/YOLO.keras/dataset/train_val/','/home/media/Documents/YOLO.keras/utils/image_name',1,1)
    #labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    #img = image(side=7,imgPath='/home/media/Documents/YOLO.keras/dataset/train_val/JPEGImages/000011.jpg')
    #img.parseXML(xmlPath='/home/media/Documents/YOLO.keras/dataset/VOCdevkit/VOC2007/Annotations/000011.xml',labels=labels,side=7)
    print boxes
