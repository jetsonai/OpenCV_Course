from Utils.wrapper import *
from Utils.utils import *


'''
def videoProcessing(openpath, savepath = None):
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
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            output = imageProcessing(frame)
            if out is not None:
                out.write(output)
            cv2.imshow("Input", frame)
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
'''


data = get_single_video()


videoProcessing(data, "output.avi")











