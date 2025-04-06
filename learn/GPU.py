#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/24 21:22
# @File ：GPU.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
# import tensorflow as tf
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     print(f"可用的GPU设备: {gpus}")


import tensorflow as tf
print("TF 版本:", tf.__version__)
# print(tf.device)
print("GPU 是否可用:", tf.config.list_physical_devices('GPU'))
# print("CUDA 版本:", tf.sysconfig.get_build_info()['cuda_version'])
# print("cuDNN 版本:", tf.sysconfig.get_build_info()['cudnn_version'])