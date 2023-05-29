from Utils.wrapper import *
from Utils.utils import *

'''
def drawPolygon(image, pts, isClosed, color=(255, 0, 0), thickness=3, lineType=cv2.LINE_AA):
    result = imageCopy(image)
    pts = pts.reshape((-1, 1, 2))
    return cv2.polylines(result, [pts], isClosed, color, thickness, lineType)
'''

data = get_single_image()
image = imageRead(data)
imageShow('image', image)

pt1 = (422,391)
pt2 = (482,392)
pt3 = (501,408)
pt4 = (504,444)
pt5 = (458,455)
pt6 = (390,454)
pt7 = (392,419)
pt8 = (405,411)
pts = np.vstack((pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8)).astype(np.int32)
pts_roi = np.array([[pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8]], dtype=np.int32)

poly_01 = drawPolygon(image, pts, False, (0, 0, 255), 5)
poly_02 = drawPolygon(image, pts, True, (0, 0, 255), 5)
poly_03 = cv2.fillPoly(image, pts_roi, (0, 0, 255))

imageShow("poly_01", poly_01)
imageShow("poly_02", poly_02)
imageShow("poly_03", poly_03)

cv2.destroyAllWindows()
