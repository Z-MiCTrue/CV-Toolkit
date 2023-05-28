import numpy as np
import cv2


#  缩放
def img_resize(img: np.ndarray, size: tuple, keep_ratio=True, points: list = None):
    """
    size=(w, h) tuple
    points:[[x_1, y_1, ...],] list
    """
    h_ori, w_ori, channel = img.shape[:3]
    w_new, h_new = size
    # 需要补边(右下补)
    if keep_ratio and w_new / w_ori != h_new / h_ori:
        scale = min(w_new / w_ori, h_new / h_ori)
        w_valid, h_valid = round(w_ori * scale), round(h_ori * scale)
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        # 补边填充
        aim_size = np.zeros((h_new, w_new, channel), dtype=np.uint8)
        h_padding = np.abs(h_new - h_valid) // 2
        w_padding = np.abs(w_new - w_valid) // 2
        aim_size[h_padding: h_valid + h_padding, w_padding: w_valid + w_padding] = img
        # 点转换
        if points is None:
            return aim_size
        else:
            points = np.array(points)
            points = points * np.tile(np.array([scale, scale]), points.shape[-1] // 2) + \
                     np.tile(np.array([w_padding, h_padding]), points.shape[-1] // 2)
            return aim_size, points.tolist()
    # 不需要改变
    elif w_new == w_ori and h_new == h_ori:
        if points is None:
            return img
        else:
            return img, points
    # 不需成比例或已成比例
    else:
        aim_size = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        if points is None:
            return aim_size
        else:
            fx = w_new / w_ori
            fy = h_new / h_ori
            points = np.array(points)
            points = points * np.tile(np.array([fx, fy]), points.shape[-1] // 2)
            return aim_size, points.tolist()


#  模板匹配
def template_match(img: np.ndarray, template, mask=None):
    if len(img.shape) >= 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if len(template.shape) >= 3:
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED, mask)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)  # 返回最大最小值及索引
    (x_1, y_1) = maxLoc
    x_2 = x_1 + template.shape[1]
    y_2 = y_1 + template.shape[0]
    return x_1, y_1, x_2, y_2


#  特征匹配
def feature_match(img_1: np.ndarray, img_2: np.ndarray, numbers=None):
    # 到目前为止，我们已经导入了要使用的模块，并定义了我们的两个图像，即模板（img_1）和用于搜索模板的图像（img_2）
    # orb = cv2.ORB_create()
    sift = cv2.SIFT_create()

    # 这是我们打算用于特征的检测器
    # kp_1, des_1 = orb.detectAndCompute(img_1,None)
    # kp_2, des_2 = orb.detectAndCompute(img_2,None)
    kp_1, des_1 = sift.detectAndCompute(img_1, None)
    kp_2, des_2 = sift.detectAndCompute(img_2, None)

    # 在这里，我们使用orb探测器找到关键点和他们的描述符。
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # 交叉最佳匹配

    # 这就是我们的BFMatcher对象
    matches = bf.match(des_1, des_2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 这里我们创建描述符的匹配，然后根据它们的距离对它们排序
    matches = matches[:numbers]
    img_3 = cv2.drawMatches(img_1, kp_1, img_2, kp_2, matches, None, flags=2)
    return matches, img_3


# canny 边缘检测
def canny_manage(img_ori: np.ndarray):
    # 高斯模糊后再边缘检测
    img_ori = cv2.GaussianBlur(img_ori, ksize=(3, 3), sigmaX=0, sigmaY=0)  # sigma值0为自动选择值
    img_canny = cv2.Canny(img_ori, threshold1=100, threshold2=200)  # threshold1:间断边缘的连接, threshold2:检测明显的边缘
    return img_canny


if __name__ == '__main__':
    pass
