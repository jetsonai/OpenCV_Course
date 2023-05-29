from Utils.wrapper import *
from Utils.utils import *

'''
def cutRectROI(image, x1, y1, x2, y2):
    return image[y1:y2, x1:x2]
def pasteRectROI(image, x1, y1, dst):
    y2, x2 = image.shape[:2]
    dst[y1:y1+y2, x1:x1+x2]=image
    return dst
def makeBlackImage(image, color=False):
    height, width = image.shape[0], image.shape[1]
    if color is True:
        return np.zeros((height, width, 3), np.uint8)
    else:
        if len(image.shape) == 2:
            return np.zeros((height, width), np.uint8)
        else:
            return np.zeros((height, width, 3), np.uint8)
def fillPolyROI(image, points):
    if len(image.shape) == 2:
        channels = 1
    else:
        channels = image.shape[2]
    mask = makeBlackImage(image)
    ignore_mask_color = (255,) * channels
    cv2.fillPoly(mask, points, ignore_mask_color)
    return mask
def polyROI(image, points):
    mask = fillPolyROI(image, points)
    return cv2.bitwise_and(image, mask)
'''

data = get_single_image()
image = imageRead(data)
imageShow('image', image)

roi_x1 = 600
roi_y1 = 362
roi_x2 = 756
roi_y2 = 489
roi_rect = cutRectROI(image, roi_x1, roi_y1, roi_x2, roi_y2)
imageShow("roi_rect", roi_rect)

image2 = imageCopy(image)
roi_new_x1 = 935
roi_new_y1 = 397
image2 = pasteRectROI(roi_rect, roi_new_x1, roi_new_y1, image2)
imageShow("image2", image2)

roi_poly_01 = np.array([[(422,391),(482,392),(501,408),(504,444),(458,455),(390,454),(392,419)]], dtype=np.int32)
image_polyROI_01 = polyROI(image, roi_poly_01)
imageShow("image_polyROI_01", image_polyROI_01)

pt1 = (1049, 370) 
pt2 = (1252, 365)
pt3 = (1253, 487)
pt4 = (1051, 486)
roi_poly_02 = np.array([[pt1, pt2, pt3, pt4]], dtype=np.int32)
image_polyROI_02 = polyROI(image, roi_poly_02)
imageShow("image_polyROI_02", image_polyROI_02)

cv2.destroyAllWindows()
