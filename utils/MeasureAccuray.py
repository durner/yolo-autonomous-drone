from keras.models import model_from_json
import theano.tensor as T

from utils.readImgFile import readImg
from utils.crop import crop_detection
from utils.ReadPascalVoc2 import prepareBatch

import os
import numpy as np

def Acc(imageList,model,sample_number=5000,thresh=0.3):
    correct = 0
    object_num = 0

    count = 0
    for image in imageList:
	count += 1
        #Get prediction from neural network
        img = crop_detection(image.imgPath,new_width=448,new_height=448)
        img = np.expand_dims(img, axis=0)
        out = model.predict(img)
        out = out[0]

        for i in range(49):
            preds = out[i*25:(i+1)*25]
            if(preds[24] > thresh):
                object_num += 1
            	row = i/7
            	col = i%7
                '''
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
                '''
            	class_num = np.argmax(preds[4:24])

                #Ground Truth
                box = image.boxes[row][col]
                if(box.has_obj):
                    for obj in box.objs:
                        true_class = obj.class_num
                        if(true_class == class_num):
                            correct += 1
	                    break


    return correct*1.0/object_num

def Recall(imageList,model,sample_number=5000,thresh=0.3):
    correct = 0
    obj_num = 0
    count = 0
    for image in imageList:
        count += 1
        #Get prediction from neural network
        img = crop_detection(image.imgPath,new_width=448,new_height=448)
        img = np.expand_dims(img, axis=0)
        out = model.predict(img)
        out = out[0]
        #for each ground truth, see we have predicted a corresponding result
        for i in range(49):
            preds = out[i*25:i*25+25]
            row = i/7
            col = i%7
            box = image.boxes[row][col]
            if(box.has_obj):
                for obj in box.objs:
                    obj_num += 1
                    true_class = obj.class_num
                    #see what we predict
                    if(preds[24] > thresh):
                        predcit_class = np.argmax(preds[4:24])
                        if(predcit_class == true_class):
                            correct += 1
    return correct*1.0/obj_num

def MeasureAcc(model,sample_number,vocPath,imageNameFile):
    imageList = prepareBatch(0,sample_number,imageNameFile,vocPath)
    acc = Acc(imageList,model)
    re = Recall(imageList,model)

    return acc,re
