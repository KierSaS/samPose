import cv2
import numpy as np

# 读取图像
depth_img = cv2.imread('/home/wubin/code/SamPose/depth/000040-depth.png', cv2.IMREAD_UNCHANGED)

# 应用均值滤波（平滑）
smoothed_img = cv2.GaussianBlur(depth_img, (5, 5), 0)

# 应用锐化滤波器
sharpening_kernel = np.array([
    [0, -1, 0],
    [-1, 5,-1],
    [0, -1, 0]
])
sharpened_img = cv2.filter2D(smoothed_img, -1, sharpening_kernel)

# 保存结果
cv2.imwrite('sharpened_depth.png', sharpened_img)
