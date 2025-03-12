#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/1/3 20:48
# @File ：test.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
import numpy as np

# 模拟输入数据 data 和 imfs
data = np.array([1, 2, 3, 4, 5])  # 一维数组，模拟原始数据
imfs = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12],
                 [13, 14, 15]])  # 二维数组，模拟固有模态函数

# 初始化去噪后的数据
denoise_data = 0

# 遍历 imfs 的每一列
for i in range(imfs.shape[1]):
    print(f"第 {i} 次迭代：")
    print(f"当前 denoise_data 的值：{denoise_data}")
    print(f"要累加的 imfs[:,{i}] 的值：{imfs[:,i]}")
    denoise_data += imfs[:,i]
    print(f"累加后的 denoise_data 的值：{denoise_data}")
    print("-" * 20)

print("最终的 denoise_data 的值：", denoise_data)