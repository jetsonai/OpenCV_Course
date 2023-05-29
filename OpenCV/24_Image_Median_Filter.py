from Utils.wrapper import *
from Utils.utils import *

'''
def imageMedianBlur(image, size):
    ksize = (size+1) * 2 - 1
    return cv2.medianBlur(image, ksize)
'''

def nothing(x):
    pass


data = get_single_image("trafficlight_only_images")
image = imageRead(data)
backup = imageCopy(image)
cv2.namedWindow('image', cv2.WINDOW_GUI_EXPANDED)

cv2.createTrackbar('BlurSize', 'image', 0, 10, nothing)

switch = '0:OFF\n1:On'
cv2.createTrackbar(switch, 'image', 1, 1, nothing)

while True:
    cv2.imshow('image', image)

    if cv2.waitKey(100) & 0xFF == 27:
        break
    size = cv2.getTrackbarPos('BlurSize', 'image')
    s = cv2.getTrackbarPos(switch, 'image')

    if s == 1:
        image = imageMedianBlur(backup, size)
    else:
        image = backup
    

cv2.destroyAllWindows()
