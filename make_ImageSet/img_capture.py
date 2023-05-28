import os

import cv2


class Options:
    def __init__(self):
        self.camera_number = 1
        self.img_dir = "./images/"
        self.width = 320
        self.height = 240
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)


if __name__ == '__main__':
    # parameters initialization
    options = Options()
    ret = False
    frame = None
    counters = 0
    vc = cv2.VideoCapture(options.camera_number, cv2.CAP_DSHOW)  # cv2.CAP_DSHOW: DirectShow 通常免驱
    vc.set(cv2.CAP_PROP_FRAME_WIDTH, options.width)
    vc.set(cv2.CAP_PROP_FRAME_HEIGHT, options.height)
    # check vc
    if vc.isOpened():
        ret, frame = vc.read()
        cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)  # AUTOSIZE <-> NORMAL
    # start
    while ret:
        cv2.imshow('frame', frame)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:  # ESC out
            cv2.destroyAllWindows()
            vc.release()
            break
        elif k in (83, 115):  # 'S' or 's' to shot
            img_name = options.img_dir + str(counters).zfill(5) + ".jpg"  # choose serial length
            cv2.imwrite(img_name, frame)
            print(img_name, ' ', "write over")
            counters += 1
        else:
            ret, frame = vc.read()
