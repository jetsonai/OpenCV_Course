from Utils.wrapper import *
from Utils.utils import *


'''
def drawLine(image, point1, point2, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA):
    result = imageCopy(image)
    return cv2.line(result, point1, point2, color, thickness, lineType)
'''


data = get_single_image()
image = imageRead(data)
imageShow('image', image)

pt1 = (600, 362)
pt2 = (756, 362)
pt3 = (756, 489)
pt4 = (600, 489)

line = drawLine(image, pt1, pt2, (0, 0, 255), 5)
line = drawLine(line, pt2, pt3, (0, 0, 255), 5)
line = drawLine(line, pt3, pt4, (0, 0, 255), 5)
line = drawLine(line, pt4, pt1, (0, 0, 255), 5)

imageShow('line', line)

cv2.destroyAllWindows()





