#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy
import numpy as np
from matplotlib import pyplot as plt



"""
直方图均衡化
输入参数
    img: 输入的图片, 要求归一化
    maxvalue: 均衡化的范围(采样精度)
返回值
    直方图均衡化的结果(归一化)
"""
def equalizeHist(img, max_value=10000):
    img1 = np.array(img*max_value, dtype=np.uint64)
    hist,bins = np.histogram(img1.flatten(),max_value+1,[0,max_value])
    hist = hist / np.sum(hist)
    hist = np.cumsum(hist)
    return hist[img1]

"""
相机响应函数使用 gamma 2.2
返回混合的两张 bgr 图象(未经tone mapping 的, 已经 tone mapping 的)
输入参数：
    ldr, intensity_map: 输入的两张图片
    tau: 权重函数的参数调节
    one_plus_weight: 在 intensity map 前权值设为 (1-weight) 还是 1
    open_equalize_hist: 是否开启直方图均衡化
    open_equalize_hist_high: 是否开启高精度采样的直方图均衡化
    only_bright_hdr: 是否只使用亮处的 intensity map
"""
def genrerate_hdr_from_ldr_and_intensity_map(ldr, intensity_map,
        tau=0.9, one_plus_weight=False, open_equalize_hist=False, open_equalize_hist_high=False,
        only_bright_hdr=False):
    # 常量
    global tau_minus_half
    global one_minus_tau
    tau_minus_half = tau - 0.5
    one_minus_tau = 1 - tau

    # + Neuromorphic Camera Guided High Dynamic Range Imaging
    # ##  Step 01: Color space conversion

    # 读入 ldr 的图片, 输入就是
    # ldr = cv2.imread(ldr_path)

    # 读取 intensity map, 输入就是
    # intensity_map = np.load(intensity_map_path)

    # + 通过相机响应函数 CRF 将 RGB 转换回线性空间
    # + 但是我们不知道 CRF 也不知道曝光时间
    # + 我们简单的使用 $L^{gamma}, gamma = \frac{1}{2.2}$ 来进行变换
    #   + 逆变换

    # CRF 逆变换
    # uint8 => uint8
    def reverse_crf(x):
        return np.round(255*((x/255)**2.2))

    ldr_linear = reverse_crf(ldr)
    ldr_linear = np.array(ldr_linear, dtype=np.uint8)

    # RGB 转为 YUV
    ldr_yuv = cv2.cvtColor(ldr_linear, cv2.COLOR_BGR2YUV)

    # ## Step 02: Spatial upsampling

    # + 将 intensity map 上采样到和图片一样的大小
    #   + 使用双三次插值

    # 变换矩阵
    M = np.float32([
        [ldr.shape[0]/intensity_map.shape[0], 0,                                   0],
        [0,                                   ldr.shape[1]/intensity_map.shape[1], 0]
    ])

    intensity_map_upsampling = cv2.warpAffine(intensity_map, M, (ldr.shape[1], ldr.shape[0]), flags=cv2.INTER_CUBIC)

    # 是否开启直方图均衡化
    if(open_equalize_hist):
        intensity_map_upsampling = cv2.equalizeHist(np.array(intensity_map_upsampling*255, dtype=np.uint8))/255
    if(open_equalize_hist_high):
        intensity_map_upsampling = equalizeHist(intensity_map_upsampling/intensity_map_upsampling.max())

    # ## Step 03: Luminance fusion

    # + 使用论文中定义的权重函数
    # + $w_i=\frac{0.5-max(|I_i-0.5|,\tau-0.5)}{1-\tau}$

    # 取出 y 轴(归一化)
    ldr_y = ldr_yuv[:,:,0].copy()/255

    # 亮度融合
    weight_matrix = []
    if(only_bright_hdr):
        weight_matrix = (0.5-np.maximum(ldr_y-0.5, np.ones_like(ldr_y)*tau_minus_half))/one_minus_tau
    else:
        weight_matrix = (0.5-np.maximum(np.abs(ldr_y-0.5), np.ones_like(ldr_y)*tau_minus_half))/one_minus_tau
    ldr_fusion = None
    if(one_plus_weight):
        ldr_fusion = weight_matrix*ldr_y + (1-weight_matrix)*intensity_map_upsampling
    else:
        ldr_fusion = weight_matrix*ldr_y + intensity_map_upsampling

    # 归一化
    ldr_fusion /= np.max(ldr_fusion)

    # ## Step 04: Chrominance compensation
    # + 将获得的图片转化为 RGB 通道, 并且进行 tone-mapping

    # 合并通道
    fusion_yuv = ldr_yuv.copy()
    fusion_yuv[:,:,0] = ldr_fusion*255

    # YUV => RGB
    fusion_bgr = cv2.cvtColor(fusion_yuv, cv2.COLOR_YUV2BGR)

    # tone mapping
    # float => uint8
    def crf(x):
        return np.round(255*((x/255)**(1/2.2)))

    fusion_tone_mapped = np.array(crf(fusion_bgr), dtype=np.uint8)

    return fusion_bgr, fusion_tone_mapped