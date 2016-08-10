import numpy as np
import os

def readImg(imgPath,h=224,w=224):
    dt = np.dtype("float32")
    testArray = np.fromfile(imgPath,dtype=dt)

    image = np.reshape(testArray,[3,h,w])
    return image
