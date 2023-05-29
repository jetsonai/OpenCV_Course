from Utils.wrapper import *
from Utils.utils import *

'''
def drawCircle(image, center, radius, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA):
    result = imageCopy(image)
    return cv2.circle(result, center, radius, color, thickness, lineType)
'''

data = get_single_image()
image = imageRead(data)
imageShow('image', image)

center_01 = (678, 418)
center_02 = (1060, 450)
center_03 = (1128, 468) 
radius = 32

circle_01 = drawCircle(image, center_01, radius, (0, 0, 255), 5)
circle_02 = drawCircle(image, center_02, radius, (0, 255, 255), 0)
circle_03 = drawCircle(image, center_03, radius, (0, 255, 0), -1)

imageShow('circle_01', circle_01)
imageShow('circle_02', circle_02)
imageShow('circle_03', circle_03)

cv2.destroyAllWindows()
