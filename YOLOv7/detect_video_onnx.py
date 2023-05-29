import cv2
import numpy as np
import onnxruntime as ort
import random
import time
import argparse


def imageProcessingCV(src_image):
    dst_image = np.copy(src_image)
    return dst_image


class DetectClass():
    def __init__(self, opt):
        weights,  imgsz = opt.weights, opt.img_size
        # Initialize
        session = ort.InferenceSession(weights, providers=['CPUExecutionProvider'])
        
        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        
        names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
            'hair drier', 'toothbrush']
        colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}
        self.names = names
        self.colors = colors
        self.session = session
        self.outname = outname
        self.inname = inname
        self.imgsz = imgsz


    def letterbox(self, im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints
        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, r, (dw, dh)


    def preProcessing(self, cv_img):
        imgsz = self.imgsz
        height, width = cv_img.shape[:2]            
        drawboard = np.zeros((height, width, 3), np.uint8)
        whiteboard = np.ones((height, width, 3), np.uint8) * 255
        image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        image, ratio, dwdh = self.letterbox(image, new_shape=(imgsz,imgsz), auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        return im, drawboard, whiteboard, ratio, dwdh
    
    
    def inference(self, im):
        outname = self.outname
        inname = self.inname
        session = self.session
        outputs = session.run(outname, {inname[0]:im})[0]
        return outputs
    
    
    def postProcessing(self, outputs, dst_image, drawboard, whiteboard, ratio, dwdh):
        # Process detections
        names = self.names
        colors = self.colors
        det_info = {}
        for _, x0, y0, x1, y1, cls_id, score in outputs:
            box = np.array([x0,y0,x1,y1])
            box -= np.array(dwdh*2)
            box /= ratio
            box = box.round().astype(np.int32).tolist()
            cls_id = int(cls_id)
            score = round(float(score),3)
            name = names[cls_id]
            if name in det_info:
                det_info[name] += 1
            else:
                det_info[name] = 1
            color = colors[name]
            name += ' '+str(score)
            cv2.rectangle(dst_image,box[:2],box[2:],color,2)
            cv2.putText(dst_image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2) 
        result = cv2.bitwise_and(dst_image, whiteboard) + drawboard
        return result, det_info
    
    
    def imageProcessing(self, cv_img):
        start = time.time()
        im, drawboard, whiteboard, ratio, dwdh = self.preProcessing(cv_img)
        
        outputs = self.inference(im)
        dst_image = imageProcessingCV(cv_img)
        
        im0, det_info = self.postProcessing(outputs, dst_image, drawboard, whiteboard, ratio, dwdh)
        
        s=''
        for k, v in det_info.items():
            s += f"{v} {k}{'s' * (v > 1)}, "  # add to string
        end = time.time() - start
        print(f'{s}Done. ({(1E3 * (end)):.1f}ms) Inference')
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
    parser.add_argument('--weights', type=str, default='yolov7-tiny.onnx', help='model.onnx path')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--source', type=str, default='../Data/blackbox_videos/video_02.avi', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    opt = parser.parse_args()
    print(opt)
    detectClass = DetectClass(opt)
    detectClass(opt.source)