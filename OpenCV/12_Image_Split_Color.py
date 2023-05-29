from Utils.wrapper import *
from Utils.utils import *

'''
def rangeColor(image, lower, upper):
    result = imageCopy(image)
    return cv2.inRange(result, lower, upper)
def splitColor(image, lower, upper):
    result = imageCopy(image)
    mask = rangeColor(result, lower, upper)
    return cv2.bitwise_and(result, result, mask=mask)
'''

data = get_single_image()
image = imageRead(data)
imageShow('image', image)

lower_white_hsv = np.array([0, 0, 100])
upper_white_hsv = np.array([180, 14, 255])
lower_yellow_hsv = np.array([10, 70, 50])
upper_yellow_hsv = np.array([22, 255, 255])

image_hsv = convertColor(image, cv2.COLOR_BGR2HSV)

white_hsv_region = rangeColor(image_hsv, lower_white_hsv, upper_white_hsv)
yellow_hsv_region = rangeColor(image_hsv, lower_yellow_hsv, upper_yellow_hsv)
output_hsv_region = white_hsv_region + yellow_hsv_region

white_hsv_overlay = splitColor(image_hsv, lower_white_hsv, upper_white_hsv)
yellow_hsv_overlay = splitColor(image_hsv, lower_yellow_hsv, upper_yellow_hsv)
output_hsv_overlay = white_hsv_overlay + yellow_hsv_overlay

imageShow("output_hsv_region", output_hsv_region)
imageShow("output_hsv_overlay", convertColor(output_hsv_overlay, cv2.COLOR_HSV2BGR))

lower_white_hls = np.array([0, 100, 0])
upper_white_hls = np.array([180, 255, 10])
lower_yellow_hls = np.array([10, 0, 10])
upper_yellow_hls = np.array([22, 255, 255])

image_hls = convertColor(image, cv2.COLOR_BGR2HLS)

white_hls_region = rangeColor(image_hls, lower_white_hls, upper_white_hls)
yellow_hls_region = rangeColor(image_hls, lower_yellow_hls, upper_yellow_hls)
output_hls_region = white_hls_region + yellow_hls_region

white_hls_overlay = splitColor(image_hls, lower_white_hls, upper_white_hls)
yellow_hls_overlay = splitColor(image_hls, lower_yellow_hls, upper_yellow_hls)
output_hls_overlay = white_hls_overlay + yellow_hls_overlay

imageShow("output_hls_region", output_hls_region)
imageShow("output_hls_overlay", convertColor(output_hls_overlay, cv2.COLOR_HLS2BGR))

cv2.destroyAllWindows()
