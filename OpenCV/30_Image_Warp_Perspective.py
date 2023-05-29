from Utils.wrapper import *
from Utils.utils import *


'''
def imagePerspectiveTransformation(image, src_pts, dst_pts, size=None, flags=cv2.INTER_LANCZOS4):
    if size is None:
        rows, cols = image.shape[:2]
        size = (cols, rows)
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return cv2.warpPerspective(image, M, dsize=size, flags=flags)
'''


data = get_single_image("lanedetection_images", index=4)
image = imageRead(data)
height, width = image.shape[:2]

src_pt1 = [626, 519]
src_pt2 = [762, 519]
src_pt3 = [905, 594]
src_pt4 = [537, 594]
dst_pt1 = [int(width*0.4), int(height*0.7)]
dst_pt2 = [int(width*0.6), int(height*0.7)]
dst_pt3 = [int(width*0.6), int(height*0.9)]
dst_pt4 = [int(width*0.4), int(height*0.9)]

src_pts = np.float32([src_pt1, src_pt2, src_pt3, src_pt4])
dst_pts = np.float32([dst_pt1, dst_pt2, dst_pt3, dst_pt4])

roadPoint = drawCircle(image, tuple(src_pt1), 10, (255, 0, 0), -1)
roadPoint = drawCircle(roadPoint, tuple(src_pt2), 10, (0, 255, 0), -1)
roadPoint = drawCircle(roadPoint, tuple(src_pt3), 10, (0, 0, 255), -1)
roadPoint = drawCircle(roadPoint, tuple(src_pt4), 10, (255, 255, 0), -1)

roadAffine_01 = imagePerspectiveTransformation(roadPoint, src_pts, dst_pts)
roadAffine_02 = imagePerspectiveTransformation(roadAffine_01, src_pts, dst_pts, flags=cv2.WARP_INVERSE_MAP)
roadAffine_03 = imagePerspectiveTransformation(roadAffine_01, dst_pts, src_pts)

imageShow("image", image)
imageShow("roadPoint", roadPoint)
imageShow("roadAffine_01", roadAffine_01)
imageShow("roadAffine_02", roadAffine_02)
imageShow("roadAffine_03", roadAffine_03)

cv2.destroyAllWindows()
