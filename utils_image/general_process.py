import numpy as np
import cv2


# 旋转
def img_rotate(img: np.ndarray, width: int = None, height: int = None, points: list = None, angle=5):
    """
    points: input: [[x, y],] -> output: [[x, y],]
    """
    if width is None or height is None:
        height, width = img.shape[:2]
    # 旋转矩阵
    mat_rotate = cv2.getRotationMatrix2D(center=(width / 2, height / 2), angle=angle, scale=1)  # 2x3
    img_rotated = cv2.warpAffine(img, mat_rotate, (width, height), borderValue=0)
    if points is None:
        return img_rotated
    else:
        points = np.array(points, dtype=np.int32)
        points_rotated = np.ones((3, points.shape[0]), dtype=np.int32)
        points_rotated[:2] = points.T
        points_rotated = np.dot(mat_rotate, points_rotated)
        points_rotated = points_rotated[:2].T.tolist()
        return img_rotated, points_rotated


# 添加高斯噪声
def img_addGauss(img, width: int = None, height: int = None, channel: int = None, scale=5e-1):
    if width is None or height is None or channel is None:
        height, width, channel = img.shape[:3]
    gauss_noise = np.random.normal(loc=0, scale=scale, size=(height, width, channel)).astype(np.uint8)
    img_noisy = cv2.add(img, gauss_noise)
    return img_noisy


# 明暗调整
def img_expose(img, degree=-30):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_hsv[:, :, 2] = cv2.add(img_hsv[:, :, 2], int(degree))
    img_hsv = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    return img_hsv


# canny 边缘检测
def canny_manage(img_ori: np.ndarray):
    # 高斯模糊后再边缘检测
    img_ori = cv2.GaussianBlur(img_ori, ksize=(3, 3), sigmaX=0, sigmaY=0)  # sigma值0为自动选择值
    img_canny = cv2.Canny(img_ori, threshold1=100, threshold2=200)  # threshold1:间断边缘的连接, threshold2:检测明显的边缘
    return img_canny


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
    # 定义特征检测器
    # orb = cv2.ORB_create()
    sift = cv2.SIFT_create()

    # kp_1, des_1 = orb.detectAndCompute(img_1,None)
    # kp_2, des_2 = orb.detectAndCompute(img_2,None)
    kp_1, des_1 = sift.detectAndCompute(img_1, None)
    kp_2, des_2 = sift.detectAndCompute(img_2, None)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # 交叉最佳匹配

    # BFMatcher对象
    matches = bf.match(des_1, des_2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 创建描述符的匹配, 并根据它们的距离对它们排序
    matches = matches[:numbers]
    img_3 = cv2.drawMatches(img_1, kp_1, img_2, kp_2, matches, None, flags=2)
    return matches, img_3


if __name__ == '__main__':
    pass
