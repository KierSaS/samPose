import numpy as np
import cv2

# 给定的pose1, pose2, 和相机内参K1, K2
pose1 = np.array([
    [-2.938309907913208008e-01, 9.358270168304443359e-01, -1.946559995412826538e-01, 1.791024506092071533e-01],
    [7.205849885940551758e-01, 8.306439965963363647e-02, -6.883730292320251465e-01, -1.442911773920059204e-01],
    [-6.280289888381958008e-01, -3.425309956073760986e-01, -6.987500190734863281e-01, 9.947724938392639160e-01],
    [0, 0, 0, 1]  # 添加齐次坐标行
])

pose2 = np.array([
    [6.673449873924255371e-01 ,7.292360067367553711e-01 ,-1.512120068073272705e-01 ,2.052416205406188965e-01],
    [4.322099983692169189e-01 ,-5.445709824562072754e-01 ,-7.187749743461608887e-01 ,-1.927561685442924500e-02],
    [-6.065019965171813965e-01 ,4.143149852752685547e-01 ,-6.786000132560729980e-01 ,1.005313873291015625e+00],
    [0, 0, 0, 1]  # 添加齐次坐标行
])

K1 = np.array([
    [572.4113999999999578, 0.000000000000000000, 325.26109999999999990],
    [0.000000000000000000, 573.5704299999999876, 242.0489900000000034],
    [0.000000000000000000, 0.000000000000000000, 1.000000000000000000]
])

K2 = np.array([
    [5.724113999999999578e+02, 0.000000000000000000e+00 ,3.252610999999999990e+02],
    [0.000000000000000000e+00, 5.735704299999999876e+02 ,2.420489900000000034e+02],
    [0.000000000000000000e+00 ,0.000000000000000000e+00 ,1.000000000000000000e+00]
])

# 计算 RT = pose2 * pose1^-1
pose1_inv = np.linalg.inv(pose1)
RT = pose2 @ pose1_inv

# 读取深度图像和RGB图像
depth_img = cv2.imread('depth/000040-depth.png', cv2.IMREAD_UNCHANGED)
rgb_img = cv2.imread('depth/000040-color.png')

# 获取图像尺寸
height, width = depth_img.shape

# 生成像素坐标
u, v = np.meshgrid(np.arange(width), np.arange(height))
homogeneous_pixel_coords = np.stack((u, v, np.ones_like(u)), axis=-1)

# 将像素坐标转换为相机1坐标系中的三维点
depth_values = depth_img.astype(float)
camera1_coords = np.linalg.inv(K1) @ homogeneous_pixel_coords.reshape(-1, 3).T
camera1_coords *= depth_values.reshape(-1)

# 将相机1坐标系中的点转换到世界坐标系
world_coords = RT[:3, :3] @ camera1_coords + RT[:3, 3:]

# 将世界坐标系中的点转换到相机2坐标系
homogeneous_world_coords = np.vstack((world_coords, np.ones((1, world_coords.shape[1]))))  # 添加齐次坐标
camera2_coords = RT[:3, :3] @ world_coords + RT[:3, 3:]

# 将相机2坐标系中的点投影回到目标图像平面
projected_pixel_coords = K2 @ camera2_coords
z = projected_pixel_coords[2]
projected_pixel_coords /= z  # 确保 z 不为零

# 重建RGB图像
projected_pixel_coords = projected_pixel_coords[:2].T.reshape(height, width, 2)
projected_pixel_coords = np.round(projected_pixel_coords).astype(int)

# 处理像素越界
projected_pixel_coords = np.clip(projected_pixel_coords, 0, [width-1, height-1])

# 创建新的RGB图像
transformed_rgb_img = np.zeros_like(rgb_img)
for i in range(height):
    for j in range(width):
        x, y = projected_pixel_coords[i, j]
        if 0 <= x < width and 0 <= y < height:
            transformed_rgb_img[i, j] = rgb_img[y, x]

# 保存新的RGB图像
cv2.imwrite('transformed_rgb.png', transformed_rgb_img)
