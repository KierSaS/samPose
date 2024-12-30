import torch
from src.matcher import Matcher, default_cfg
import os
import pandas as pd 
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import cv2
import json
import shutil
from numpy.linalg import inv
from src.utils.dataset import (
    read_scannet_gray,
    read_scannet_grayv2,
)
from tabulate import tabulate
from loguru import logger
# from exps.epipolar_util import (
#     draw_epiplor_line,
# )

from src.utils.metrics import estimate_pose, relative_pose_error
def resample_kpts(kpts: np.ndarray, height_ratio, width_ratio):
    kpts[:, 0] *= width_ratio
    kpts[:, 1] *= height_ratio
    return kpts

def _np_to_cv2_kpts(np_kpts):
    cv2_kpts = []
    for np_kpt in np_kpts:
        cur_cv2_kpt = cv2.KeyPoint()
        cur_cv2_kpt.pt =(int(np_kpt[0]),int(np_kpt[1]))
        cv2_kpts.append(cur_cv2_kpt)
    return cv2_kpts
###以上用于对关键点进行重采样和将 NumPy 格式的关键点转换为 OpenCV 格式的关键点。

if __name__ == "__main__":

    matcher = Matcher(config=default_cfg)
    matcher.load_state_dict(torch.load("/home/dlab4090/wubin/pope/weights/matcher.pth")['state_dict'], strict = False )
    matcher = matcher.eval().cuda()

    image0_name = "/home/dlab4090/wubin/pope/data/demos/inputs/prompt.png"
    image1_name = "/home/dlab4090/wubin/pope/data/demos/inputs/targetcrop.png"

    # # Ref: https://github.com/applied-ai-lab/zero-shot-pose
    # import sys
    # sys.path.insert(0, "zero-shot-pose/")
    # from zsp.method.calc_ref_to_target_pose import get_correspondance
    # mkpts0, mkpts1  = get_correspondance(image0_name, image1_name)
    # #从指定路径加载图像并获取关键点坐标 mkpts0 和 mkpts1。
    # # 这些关键点坐标是通过调用外部库中的函数 get_correspondance() 获取的。
    #
    # mkpts0  = mkpts0.cpu().numpy()
    # mkpts1  = mkpts1.cpu().numpy()
    # mkpts0  = mkpts0[:,[1,0]]
    # mkpts1  = mkpts1[:,[1,0]]
    #
    # mkpts1 = mkpts1.astype(np.float32)
    # mkpts0 = mkpts0.astype(np.float32)
    # print(mkpts0.shape, mkpts1.shape)
    # mkpts0, mkpts1 = _np_to_cv2_kpts(mkpts0), _np_to_cv2_kpts(mkpts1)
    # #将获取的关键点坐标转换为 NumPy 数组，并将其转换为 OpenCV 格式的关键点。
    #
    # img1 = cv2.resize(cv2.imread(image1_name),(512,512))
    # img0 = cv2.resize(cv2.imread(image0_name),(512,512))
    #
    # matched_image = cv2.drawMatches(
    #     img0,
    #     mkpts0,
    #     img1,
    #     mkpts1,
    #     [
    #         cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
    #         for idx in range(len(mkpts1))
    #     ],
    #     None,
    #     flags=2,
    # )
    # cv2.imwrite("match_res_1.jpg", matched_image)
    #使用 OpenCV 的 cv2.drawMatches() 函数在两个图像中绘制匹配的关键点。
    # 这里使用了匹配的索引来创建 cv2.DMatch 对象，以便将匹配的关键点连接起来，并将结果保存为 "match_res_1.jpg" 文件。
#########################################match_res_1


#######第二段测试，分别使用zeroshot和Matcher两个方法进行匹配对应点的计算；

    img0 =  read_scannet_gray(image0_name,(512,512)).unsqueeze(0).cuda()
    img1 =  read_scannet_gray(image1_name,(512,512)).unsqueeze(0).cuda()
    batch = {'image0': img0, 'image1': img1}
    #从指定路径加载图像并将其转换为模型可接受的格式
    # Inference
    with torch.no_grad():
        matcher(batch)    # batch = {'image0': img0, 'image1': img1}
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()
        confidences = batch["mconf"].cpu().numpy()
        #前向推理获取模型输出的匹配关键点坐标 mkpts0 和 mkpts1，以及对应的置信度 confidences

    conf_mask = np.where(confidences > 0)
    mkpts0 = mkpts0[conf_mask]
    mkpts1 = mkpts1[conf_mask]
    mkpts0, mkpts1 = _np_to_cv2_kpts(mkpts0), _np_to_cv2_kpts(mkpts1)
    img1 = cv2.resize(cv2.imread(image1_name),(512,512))
    img0 = cv2.resize(cv2.imread(image0_name),(512,512))

    matched_image = cv2.drawMatches(
        img0,
        mkpts0,
        img1,
        mkpts1,
        [
            cv2.DMatch(_queryIdx=idx, _trainIdx=idx, _distance=0)
            for idx in range(len(mkpts1))
        ],
        None,
        flags=2,
    )
    #再次使用 cv2.drawMatches() 函数在两个图像中绘制匹配的关键点，并将结果保存为 "match_res_2.jpg" 文件。
    cv2.imwrite("match_res_2.jpg", matched_image)


#第一部分代码使用了外部库提供的函数进行匹配，而第二部分代码则使用了一个加载的模型进行推理并得到匹配结果。

    
