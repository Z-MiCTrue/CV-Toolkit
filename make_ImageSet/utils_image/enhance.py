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


if __name__ == '__main__':
    pass
