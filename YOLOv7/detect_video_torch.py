## NEED TO MOVE THIS SCRIPT TO YOLOV7 PATH
import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized


class DetectClass():
    def __init__(self, opt):
        weights,  imgsz = opt.weights, opt.img_size
        # Initialize
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if half:
            model.half()  # to FP16
            
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        self.model = model
        self.img_size = imgsz
        self.stride = 32
        self.device = device
        self.half = half
        self.names = names
        self.colors = colors
        
        
    def preProcessing(self, cv_img):
        device = self.device
        half = self.half
        # Padded resize
        img = letterbox(cv_img, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        return img, cv_img
    
    
    @torch.no_grad()
    def inference(self, img):
        model = self.model
        t0 = time.time()
        # Inference
        t1 = time_synchronized()
        pred = model(img)[0]
        t2 = time_synchronized()
        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()
        return pred, t0, t1, t2, t3
        
        
    def postProcessing(self, pred, img, im0):# Process detections
        names = self.names
        colors = self.colors
        det_info = {}
        det = pred[0]
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class
                det_info[names[int(c)]] = n
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)
        return im0, det_info
        
        
    def imageProcessing(self, cv_img):
        img, im0 = self.preProcessing(cv_img)
        pred, t0, t1, t2, t3 = self.inference(img)
        im0, det_info = self.postProcessing(pred, img, im0)
        s=''
        for k, v in det_info.items():
            s += f"{v} {k}{'s' * (v > 1)}, "  # add to string
        print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
        return im0
            
        
    def videoProcessing(self, openpath, savepath = None):
        cap = cv2.VideoCapture(openpath)
        if cap.isOpened():
            print("Video Opened")
        else:
            print("Video Not Opened")
            print("Program Abort")
            exit()
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = None
        if savepath is not None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
        cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                output = self.imageProcessing(frame)
                if out is not None:
                    out.write(output)
                cv2.imshow("Output", output)
            else:
                break
            if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
                break
        cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()
        return


    def __call__(self, video_path):
        self.videoProcessing(video_path)
        

    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7-tiny.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--source', type=str, default='../../Data/blackbox_videos/video_02.avi', help='source')  # file/folder, 0 for webcam
    
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    opt = parser.parse_args()
    print(opt)
    detectClass = DetectClass(opt)
    detectClass(opt.source)
