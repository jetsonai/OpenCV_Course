from Utils.wrapper import *
from Utils.utils import *


'''
def drawRect(image, point1, point2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA):
    result = imageCopy(image)
    return cv2.rectangle(result, point1, point2, color, thickness, lineType)
'''


data = get_single_image()
image = imageRead(data)
imageShow('image', image)

pt1 = (600, 362)
pt2 = (756, 489)

rect_01 = drawRect(image, pt1, pt2, (0, 0, 255), 5)
rect_02 = drawRect(image, pt1, pt2, (0, 0, 255), 0)
rect_03 = drawRect(image, pt1, pt2, (0, 0, 255), -1)

imageShow('rect_01', rect_01)
imageShow('rect_02', rect_02)
imageShow('rect_03', rect_03)

cv2.destroyAllWindows()
