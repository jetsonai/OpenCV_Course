from Utils.wrapper import *
from Utils.utils import *

'''
def imageParameters(imagename, image):
    height, width = image.shape[0], image.shape[1]
    print("{}.shape is {}".format(imagename, image.shape))
    print("{}.shape[0] is height: {}".format(imagename, height))
    print("{}.shape[1] is width: {}".format(imagename, width))
    if len(image.shape) == 2:
        print("This is grayscale image.")
        print("{}.shape[2] is Not exist, So channel is 1".format(imagename))
    else:
        print("This is not grayscale image.")
        print("{}.shape[2] is channels: {}".format(imagename, image.shape[2]))
    print(" {}.dtype is {}".format(imagename, image.dtype))
    return height, width
'''


data = get_single_image()
image = imageRead(data)
height, width = imageParameters("Image", image)
imageShow("Opened Image", image)
print(height, width)






