import numpy as np
import cv2


class Common_Paras:
    def __init__(self):
        self.color_list = ['0-red', '1-green', '2-blue',
                           '3-orange', '4-yellow', '5-cyan', '6-violet']
        self.hsv_lower = np.array([[0, 43, 46], [35, 43, 46], [100, 43, 46],
                                   [11, 43, 46], [26, 43, 46], [78, 43, 46], [125, 43, 46]])
        self.hsv_upper = np.array([[10, 255, 255], [77, 255, 255], [124, 255, 255],
                                   [25, 255, 255], [34, 255, 255], [99, 255, 255], [155, 255, 255]])
        self.color_list_rgb = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
                               (0, 128, 255), (0, 255, 255), (255, 255, 0), (128, 0, 128)]
        self.color_num = len(self.color_list)


def hsv_ColorFilter(img, hsv_paras, serial, min_area=512):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(img_hsv, hsv_paras.hsv_lower[serial], hsv_paras.hsv_upper[serial])
    blur = cv2.GaussianBlur(mask, (15, 15), 0)
    kernel = np.ones((5, 5), np.uint8)
    img_closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)
    img_opening = cv2.morphologyEx(img_closing, cv2.MORPH_OPEN, kernel)
    # OpenCV 3 (返回的图像, 每个轮廓的点集, 每个轮廓对应的层次信息
    binary, contours, hierarchy = cv2.findContours(img_opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    locations = []
    for j, points in enumerate(contours):
        rect = cv2.minAreaRect(points)  # 提取矩形坐标(中心点, 长和宽)
        detect_area = np.max(rect[1]) * np.min(rect[1])
        if detect_area > min_area:
            # 贴字
            rect_x, rect_y = np.int0(rect[0])
            cv2.putText(img,
                        '({})'.format(hsv_paras.color_list[serial]),
                        (rect_x, rect_y), font, 0.5, hsv_paras.color_list_rgb[serial], 1)
            # 画框
            box = [np.int0(cv2.boxPoints(rect))]
            cv2.drawContours(img, box, 0, (0, 255, 0), 2)
            # 加坐标
            locations.append([rect_x, rect_y])
    return locations, img


if __name__ == '__main__':
    ret = False
    frame = None
    vc = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # init parameters
    common_paras = Common_Paras()
    font = cv2.FONT_HERSHEY_SIMPLEX
    if vc.isOpened():
        ret, frame = vc.read()
        cv2.namedWindow('frame', cv2.WINDOW_AUTOSIZE)  # AUTOSIZE <-> NORMAL
    # start
    while ret:
        for i in range(common_paras.color_num):
            locations_xy, frame = hsv_ColorFilter(frame, common_paras, i)
            print(locations_xy)
        cv2.imshow('frame', frame)
        k = cv2.waitKey(10) & 0xFF
        if k == 27:
            cv2.destroyAllWindows()
            vc.release()
            break
        else:
            ret, frame = vc.read()
