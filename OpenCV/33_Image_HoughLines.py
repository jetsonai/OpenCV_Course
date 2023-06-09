from Utils.wrapper import *
from Utils.utils import *


'''
def houghLines(image, rho=1, theta=np.pi/180, threshold=100):
    return cv2.HoughLines(image, rho, theta, threshold)
def drawHoughLines(image, lines):
    result = imageCopy(image)
    if len(image.shape) == 2:
        result = convertColor(image, cv2.COLOR_GRAY2BGR)
    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)
    return result
'''

data = get_single_image("lanedetection_images", index=4)
image = imageRead(data)
imageShow("image", image)

image_edge = cannyEdge(image, 100, 200)
lines = houghLines(image_edge, 1, np.pi/180, 100)
image_lines = drawHoughLines(image, lines)
imageShow("image_edge", image_edge)
imageShow("image_lines", image_lines)

cv2.destroyAllWindows()
