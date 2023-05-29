from Utils.wrapper import *
from Utils.utils import *


'''
def convertColor(image, flag=cv2.COLOR_BGR2GRAY):
    return cv2.cvtColor(image, flag)
'''


data = get_single_image(index=4)
image = imageRead(data)
imageShow('image', image)
 
b, g, r = splitImage(image)
imageShow("b", b)
imageShow("g", g)
imageShow("r", r)

imageHSV = convertColor(image, cv2.COLOR_BGR2HSV)
imageShow("image.HSV", imageHSV)
h, s, v = splitImage(imageHSV)
imageShow("h", h)
imageShow("s", s)
imageShow("v", v)

imageHLS = convertColor(image, cv2.COLOR_BGR2HLS)
imageShow("imageHLS", imageHLS)
h2, l2, s2 = splitImage(imageHLS)
imageShow("h2", h2)
imageShow("l2", l2)
imageShow("s2", s2)

imageGray = convertColor(image, cv2.COLOR_BGR2GRAY)
imageShow("imageGray", imageGray)

cv2.destroyAllWindows()
