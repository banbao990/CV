#!/usr/bin/env python
# coding: utf-8

import utils as my_util
import os
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="HDR using a LDR image with an intensity map")
# args
parser.add_argument("--img-dir", default="data/your_cases", help="diretory of images")

args = parser.parse_args()


def main(dir_path):

    # 打开文件夹
    image_dir = os.listdir(dir_path)

    for f in image_dir:
        ldr_path = "{}/{}/ldr.jpg".format(dir_path, f)
        intensity_map_path = "{}/{}/intensity_map.npy".format(dir_path, f)
        non_tone_mapped_path = "{}/{}/result(non-tonemapped)".format(dir_path, f)
        tone_mapped_path = "{}/{}/result".format(dir_path, f)
        
        # 读取图片
        ldr = cv2.imread(ldr_path)
        intensity_map =np.load(intensity_map_path)
        intensity_map /= intensity_map.max()
        
        # 保存 intensity map
        cv2.imwrite("{}/{}/intensity_map.jpg".format(dir_path, f), intensity_map*255)

        # 1-w
        a, b = my_util.genrerate_hdr_from_ldr_and_intensity_map(
            ldr, intensity_map, one_plus_weight=True)
        suffix = ".1.minis.w.jpg"
        cv2.imwrite(non_tone_mapped_path + suffix, a)
        cv2.imwrite(tone_mapped_path + suffix, b)

        # w
        a, b = my_util.genrerate_hdr_from_ldr_and_intensity_map(
            ldr, intensity_map)
        suffix = ".1.jpg"
        cv2.imwrite(non_tone_mapped_path + suffix, a)
        cv2.imwrite(tone_mapped_path + suffix, b)

        # 1-w, equal(256)
        a, b = my_util.genrerate_hdr_from_ldr_and_intensity_map(
            ldr, intensity_map, one_plus_weight=True,
            open_equalize_hist=True)

        suffix = ".1.minis.w.equalize_hist(256).jpg"
        cv2.imwrite(non_tone_mapped_path + suffix, a)
        cv2.imwrite(tone_mapped_path + suffix, b)

        # 1-w, equal(10000)
        a, b = my_util.genrerate_hdr_from_ldr_and_intensity_map(
            ldr, intensity_map, one_plus_weight=True,
            open_equalize_hist_high=True)

        suffix = ".1.minis.w.equalize_hist(10000).jpg"
        cv2.imwrite(non_tone_mapped_path + suffix, a)
        cv2.imwrite(tone_mapped_path + suffix, b)

        # 1-w, equal(256), 只使用高亮度的intensity map, 舍弃暗处
        a, b = my_util.genrerate_hdr_from_ldr_and_intensity_map(
            ldr, intensity_map, one_plus_weight=True,
            open_equalize_hist=True, only_bright_hdr=True)

        suffix = ".1.minis.w.equalize_hist(256).only.bright.jpg"
        cv2.imwrite(non_tone_mapped_path + suffix, a)
        cv2.imwrite(tone_mapped_path + suffix, b)

        # 1-w, equal(10000), 只使用高亮度的intensity map, 舍弃暗处
        a, b = my_util.genrerate_hdr_from_ldr_and_intensity_map(
            ldr, intensity_map, one_plus_weight=True,
            open_equalize_hist_high=True, only_bright_hdr=True)

        suffix = ".1.minis.w.equalize_hist(10000).only.bright.jpg"
        cv2.imwrite(non_tone_mapped_path + suffix, a)
        cv2.imwrite(tone_mapped_path + suffix, b)

if __name__ == "__main__":
    main(args.img_dir)