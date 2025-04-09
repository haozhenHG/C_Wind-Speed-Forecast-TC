#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/9 16:23
# @File ：npy_.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
import numpy as np

# 从 .npy 文件中加载数组
loaded_arr = np.load('metrics.npy')

print(loaded_arr)

loaded_arr = np.load('pred.npy')

print(loaded_arr)
print(loaded_arr.shape)