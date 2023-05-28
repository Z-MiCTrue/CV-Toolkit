from threading import Thread
from copy import deepcopy
import time

import cv2


class Streamer:
    def __init__(self, cam_id):
        # def vc
        self.vc = cv2.VideoCapture(cam_id)
        # init status
        self.cam_state = False
        self.frame = None
        # creat threading
        self.thread = Thread(name='camera', target=self.update, daemon=True)  # open threading till main killed
        self.thread.start()
        print('camera threading start')
        # wait for camera to open
        time.sleep(2)

    def update(self):
        if self.vc.isOpened():
            self.cam_state = True
        while self.cam_state:
            self.cam_state, self.frame = self.vc.read()

    def grab_frame(self):
        if self.cam_state:
            return deepcopy(self.frame)
        else:
            return None


if __name__ == '__main__':
    streamer = Streamer(1)
    while True:
        frame = streamer.grab_frame()
        if frame is not None:
            cv2.imshow("frame", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:  # ESC out
                cv2.destroyAllWindows()
                break
        else:
            break
    print('over')
