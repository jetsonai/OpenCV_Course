from Utils.wrapper import *
from Utils.utils import *


'''
def imageRead(openpath, flag=cv2.IMREAD_COLOR):
    image = cv2.imread(openpath, flag)
    if image is not None:
        print("Image Opened")
        return image
    else:
        print("Image Not Opened")
        print("Program Abort")
        exit()
        

def imageShow(imagename, image, flag=cv2.WINDOW_GUI_EXPANDED):
    cv2.namedWindow(imagename, flag)
    cv2.imshow(imagename, image)
    cv2.waitKey()
    return


def imageWrite(imagename, image):
    return cv2.imwrite(imagename, image)
'''


data = get_single_image()


# cv2.IMREAD_COLOR
# Blue - Green - Red 순서의 3채널 이미지로 열림
image_color = imageRead(data, cv2.IMREAD_COLOR)

# cv2.IMREAD_GRAYSCALE
# 회색조 1채널로 열림
image_gray = imageRead(data, cv2.IMREAD_GRAYSCALE)

# cv2.IMREAD_UNCHANGED
# 1채널 회색조 이미지는 1채널로 열리고, 3채널 컬러 이미지는 3채널 BGR로 열림
image_origin = imageRead(data, cv2.IMREAD_UNCHANGED)


# cv2.WINDOW_NORMAL: 화면 크기 전환 가능 및 비율 유지안됨
imageShow("image_color, cv2.WINDOW_NORMAL", image_color, cv2.WINDOW_NORMAL)

# cv2.WINDOW_AUTOSIZE : 화면 크기 전환 불가능 및 비율 유지
imageShow("image_color, cv2.WINDOW_AUTOSIZE", image_color, cv2.WINDOW_AUTOSIZE)

# cv2.WINDOW_FREERATIO : 화면 크기 전환 가능 및 비율 유지 안됨
imageShow("image_color, cv2.WINDOW_FREERATIO", image_color, cv2.WINDOW_FREERATIO)

# cv2.WINDOW_GUI_NORMAL : 화면 크기 전환 가능 및 비율 유지 안됨
imageShow("image_color, cv2.WINDOW_GUI_NORMAL", image_color, cv2.WINDOW_GUI_NORMAL)

# cv2.WINDOW_GUI_EXPANDED : 화면 크기 전환 가능 및 비율 유지 안됨
imageShow("image_color, cv2.WINDOW_GUI_EXPANDED", image_color, cv2.WINDOW_GUI_EXPANDED)


imageWrite("gray.jpg", image_gray)