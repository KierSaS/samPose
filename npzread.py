import numpy as np

# 读取 .npz 文件
npz_file = np.load('/home/wubin/code/SamPose/data/ycbv/test_video/obj_000015/info.npz')

# 查看文件中的所有数组的名称
print(npz_file.files)

# # 访问具体的数组
array1 = npz_file['point']
# array2 = npz_file['array2_name']
#
# # 打印数组内容
print(array1)
# print(array2)
