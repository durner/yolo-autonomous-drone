import os
import xml.etree.ElementTree as ET
from crop import crop_detection
import numpy as np
from PIL import Image
from PIL import ImageDraw
import scipy
import random


vocPath = os.path.abspath(os.path.join(os.getcwd(),os.path.pardir,'dataset'))

class objInfo():
    """
    objInfo saves the information of an object, including its class num, its cords
    """
    def __init__(self,x,y,h,w,class_num):
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.class_num = class_num

class Cell():
    """
    A cell is a grid cell of an image, it has a boolean variable indicating whether there are any objects in this cell,
    and a list of objInfo objects indicating the information of objects if there are any
    """
    def __init__(self):
        self.has_obj = False
        self.objs = []

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
        self.boxes = []
        for i in range(side):
            rows = []
            for j in range(side):
                rows.append(Cell())
            self.boxes.append(rows)

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
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            h = ymax-ymin
            w = xmax-xmin
            #objif = objInfo(xmin/448.0,ymin/448.0,np.sqrt(ymax-ymin)/448.0,np.sqrt(xmax-xmin)/448.0,class_num)

            #which cell this obj falls into
            centerx = (xmax+xmin)/2.0
            centery = (ymax+ymin)/2.0
            newx = (448.0/width)*centerx
            newy = (448.0/height)*centery

            h_new = h * (448.0 / height)
            w_new = w * (448.0 / width)

            cell_size = 448.0/side
            col = int(newx / cell_size)
            row = int(newy / cell_size)
           # print "row,col:",row,col,centerx,centery

            cell_left = col * cell_size
            cell_top = row * cell_size
            cord_x = (newx - cell_left) / cell_size
            cord_y = (newy - cell_top)/ cell_size

            objif = objInfo(cord_x,cord_y,np.sqrt(h_new/448.0),np.sqrt(w_new/448.0),class_num)
            self.boxes[row][col].has_obj = True
            self.boxes[row][col].objs.append(objif)

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

                y = []
                for i in range(7):
                    for j in range(7):
                        box = image.boxes[i][j]
                        '''
                        ############################################################
                        #x,y,h,w,one_hot class label vector[0....0],objectness{0,1}#
                        ############################################################
                        '''
                        if(box.has_obj):
                            obj = box.objs[0]

                            y.append(obj.x)
                            y.append(obj.y)
                            y.append(obj.h)
                            y.append(obj.w)

                            labels = [0]*20
                            labels[obj.class_num] = 1
                            y.extend(labels)
                            y.append(1) #objectness
                        else:
                            y.extend([0]*25)
                y = np.asarray(y)
                #y = np.reshape(y,[1,y.shape[0]])

                images.append(image_array)
                boxes.append(y)
            #return np.asarray(images),np.asarray(boxes)
            yield np.asarray(images),np.asarray(boxes)

if __name__ == '__main__':
    imageNameFile='/home/media/Documents/YOLO.keras/dataset/train_val/SingleImageNameFile.txt'
    vocPath='/home/media/Documents/YOLO.keras/dataset/train_val'
    '''
    imageList = prepareBatch(0,2,imageNameFile,vocPath)
    for i in range(0,2):
        img = imageList[i]
        print img.imgPath
        boxes = img.boxes
        for i in range(7):
            for j in range(7):
                if(boxes[i][j].has_obj):
                    print i,j
                    objs = boxes[i][j].objs
                    for obj in objs:
                        print obj.class_num
                        print obj.x
                        print obj.y
                        print
    '''
    image_array,y = generate_batch_data(vocPath,imageNameFile,1,sample_number=16)
    print image_array.shape,y.shape
    #print image_array[0,...,...,...].shape
    #let's see if we read correctly
    image_array = image_array[0,...,...,...]
    #scipy.misc.imsave('recovered.jpg', image_array)
    print image_array.shape
    image_array = (image_array + 1.0) / 2.0 * 225.0
    image_array = np.rollaxis(image_array,2,0)
    image_array = np.rollaxis(image_array,2,0)
    print image_array.shape

    scipy.misc.imsave('recovered.jpg', image_array)
    # center should be in (3,3)
    labels = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train","tvmonitor"]
    out = y[0]

    imgPath = os.path.join(os.getcwd(),'recovered.jpg')
    img = Image.open(imgPath)
    image_arr,img_resize = crop_detection(imgPath,448,448,save=True)
    drawable = ImageDraw.Draw(img_resize)
    #Draw orignal bounding boxes

    count = 0
    for i in range(49):
        preds = out[i*25:(i+1)*25]
        if(preds[24] > 0.3):
            count = count + 1
            #print preds[0:4],preds[24]
            row = i/7
            col = i%7
            print row,col
            centerx = 64 * col + 64 * preds[0]
            centery = 64 * row + 64 * preds[1]

            h = preds[2] * preds[2]
            h = h * 448.0
            w = preds[3] * preds[3]
            w = w * 448.0

            left = centerx - w/2.0
            right = centerx + w/2.0
            up = centery - h/2.0
            down = centery + h/2.0

            if(left < 0): left = 0
            if(right > 448): right = 447
            if(up < 0): up = 0
            if(down > 448): down = 447

            drawable.rectangle([left,up,right,down],outline='red')
            print 'Class is: ',labels[np.argmax(preds[4:24])]
            print np.max(preds[4:24])
    print count
    img_resize.save(os.path.join(os.getcwd(),'recover.jpg'))
