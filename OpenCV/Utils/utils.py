import os
from .wrapper import *


class PathClass():
    def __init__(self):
        path, _ = os.path.split(os.path.abspath(__file__))
        self.data_root = os.path.join(path, "..", "..", "Data")
        self.data_name_list = os.listdir(self.data_root)
        # ['blackbox_images', 'blackbox_videos', 'lanedetection_videos', 'lanedetection_images', 'trafficlight_images', 'trafficlight_videos']
        
        
    def get_data(self, get_data_string):
        data_root = os.path.join(self.data_root, get_data_string)
        init_list = os.listdir(data_root)
        data_list = []
        for data_name in init_list:
            data_list.append(os.path.join(data_root, data_name))
        return data_list
        
    
    def __call__(self, get_data_string):
        if get_data_string in self.data_name_list:
            return self.get_data(get_data_string)
        else:
            print("Wrong data name")
            exit()
            
            
def get_single_data(get_data_string, index=0):
    pathClass = PathClass()
    data_list = pathClass(get_data_string)
    return data_list[index]


def get_multiple_datas(get_data_string):
    pathClass = PathClass()
    data_list = pathClass(get_data_string)
    return data_list
    
            
def get_single_image(get_data_string = 'blackbox_images', index=0):
    return get_single_data(get_data_string, index)
            
            
def get_single_video(get_data_string = 'blackbox_videos', index=0):
    return get_single_data(get_data_string, index)


def get_multiple_images(get_data_string = 'blackbox_images'):
    return get_multiple_datas(get_data_string)


def get_multiple_videos(get_data_string = 'blackbox_videos'):
    return get_multiple_datas(get_data_string)
            

def processingSingleImage(imagePath, sourceName = "Opened Image", resultName = "Result Image", outputPath = "output.jpg"):
    image = imageRead(imagePath)
    imageShow(sourceName, image)
    result = imageProcessing(image)
    imageShow(resultName, result)
    imageWrite(outputPath, result)
    return


def processingSingleVideo(videoPath, outputPath = "output.avi"):
    videoProcessing(videoPath, outputPath)
    return


def processingMultipleImages(imageList):
    for index, imagePath in enumerate(imageList):
        processingSingleImage(imagePath, "Image index is {}".format(index), "Result of index {}".format(index), "Result_{}.jpg".format(index))
    return


def processingMultipleVideos(videoList):
    for index, videoPath in enumerate(videoList):
        processingSingleVideo(videoPath, "output_{}.avi".format(index))