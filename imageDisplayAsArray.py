#
#  Created by Si Te Feng on Jul/3/16.
#  Copyright c 2016 Si Te Feng. All rights reserved.
#
#  notMNIST download code referenced from Google Inc.
#  Data parsing partially referenced from Josh Bleecher Snyder
#
#  Not to be used for commercial purposes.
#

'''
Loading an image and display the image as numpy array to identify color channels
'''

import numpy as np
from PIL import Image
import glob
from scipy import ndimage


allLetterArray = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
allLetterArrayCount = 10


# param: image path to read from
# returns: image as a python list
def loadImage(imagePath):
    try:
        imFile = Image.open(imagePath)
    except IOError as err:
        print("IO error: {0}".format(err))
        return
    except:
        print("Unexpected error: {0}".format(err))
        return

    img = list(imFile.getdata())
    return img

# param takes a numpy array
def convertToSingleChannelFromRGBAImage(image):
    useAlphaChannel = False
    for i in xrange(image[:, 3].size):
        pixelValue = image[i, 3]
        if pixelValue != 255:
            useAlphaChannel = True

    if useAlphaChannel:
        print("Using alpha channel on image")
        return image[:, 3]
    else:
        print("Using R channel on image")
        return image[:, 0]


manualImagePath = "./letterA.png"

_manualImage = np.array(loadImage(manualImagePath))
print(_manualImage.shape)

manualImage = convertToSingleChannelFromRGBAImage(_manualImage)
print(manualImage)

