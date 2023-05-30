import numpy as np
import cv2


def find_points(img):
    xy_points = np.argwhere(img > 0)
    return xy_points


def xy_hough(xy_points):
    x_mat, y_mat = np.split(xy_points.repeat(181, axis=1), 2, axis=1)
    theta = np.arange(-90, 91, 1)
    ruler_theta = np.expand_dims(theta / 180 * np.pi, axis=0)
    rho = x_mat * np.cos(ruler_theta) + y_mat * np.sin(ruler_theta)
    theta = np.expand_dims(theta, axis=0).repeat(rho.shape[0], axis=0)
    res = np.insert(theta, range(1, 182), rho, axis=1)
    res = np.concatenate(np.split(res, 181, axis=1), axis=0)
    return res  # [theta, rho]


def draw_hough(hough_points):
    # 数据偏移至正值
    bias = np.min(hough_points, axis=0)
    hough_points -= bias
    # 绘制换到霍夫空间的曲线
    hough_img = np.zeros(np.max(hough_points, axis=0)[::-1] + 1, dtype=int)
    for point in hough_points:  # 绘制霍夫空间的曲线
        hough_img[point[1], point[0]] += 1
    hough_img = cv2.normalize(hough_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)  # 归一化
    return hough_img, bias


if __name__ == '__main__':
    test_img = cv2.imread('test.png', 0)
    test_img = 255 - test_img  # 反转白底

    key_points_test = find_points(test_img)
    hough_points_test = xy_hough(key_points_test)
    hough_img_test, bias_test = draw_hough(hough_points_test)

    cv2.imshow('hough', hough_img_test)
    cv2.waitKey()
