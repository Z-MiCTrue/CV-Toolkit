import numpy as np
import cv2
import matplotlib.pyplot as plt


def fft_img(img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 快速傅里叶变换: 空域-->频域
    img_fft = np.fft.fft2(img)
    # 中心化
    img_fft = np.fft.fftshift(img_fft)
    max_fft, min_fft = np.max(np.abs(img_fft)), np.min(np.abs(img_fft))
    print('max amplitude:', max_fft)
    print('min amplitude:', min_fft)
    magnitude_spectrum = cv2.normalize(np.log(np.abs(img_fft)), None, alpha=0, beta=255,
                                       norm_type=cv2.NORM_MINMAX)  # 归一化
    return img_fft, magnitude_spectrum


def ifft_img(img_fft):
    # 去中心化
    ifft_shift = np.fft.ifftshift(img_fft)
    # 逆傅里叶变换: 频域-->空域
    img_ifft = np.fft.ifft2(ifft_shift)
    # 取模(幅值)
    img_res = np.abs(img_ifft).astype(np.uint8)
    return img_ifft, img_res


def fft_mask(img, radius, contrast_show=False):
    img_fft, magnitude_spectrum = fft_img(img)
    # 掩膜
    mask = np.zeros(img_fft.shape[:2], dtype=np.uint8)
    cv2.circle(mask, np.array(img_fft.shape[:2][::-1]) // 2, radius, 1, -1)  # -1 表示实心
    img_fft *= mask
    magnitude_spectrum_masked = magnitude_spectrum * mask
    img_ifft, img_res = ifft_img(img_fft)
    # 对比展示
    if contrast_show:
        plt.figure()
        plt.subplot(221)
        plt.imshow(img, cmap="gray")
        plt.title("Input Image", fontsize=10)
        plt.subplot(222)
        plt.imshow(magnitude_spectrum.astype(np.uint8), cmap="gray")
        plt.title("Magnitude Spectrum", fontsize=10)
        plt.subplot(223)
        plt.imshow(img_res, cmap="gray")
        plt.title("Image after Filter", fontsize=10)
        plt.subplot(224)
        plt.imshow(magnitude_spectrum_masked, cmap="gray")
        plt.title("Magnitude Spectrum (Masked)", fontsize=10)
        plt.show()
    return img_res


if __name__ == '__main__':
    binary_img = cv2.imread('rice.png', 0)
    binary_img = fft_mask(binary_img, 50, contrast_show=True)
    binary_img = cv2.adaptiveThreshold(binary_img, maxValue=255, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       thresholdType=cv2.THRESH_BINARY, blockSize=81, C=0)
    cv2.imshow('', binary_img)
    cv2.waitKey()
