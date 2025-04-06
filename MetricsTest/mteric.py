#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/4/4 16:09
# @File ：mteric.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
import numpy as np
import pandas as pd
# 1. 生成两个文件


# 生成从 1 到 100 的数组，步长为 1
test1 = np.arange(1, 101, 1)  # 生成 1 到 100 的数组
print(test1)
np.savetxt("test1.csv", test1, delimiter=',')

test2= np.arange(2, 102, 1)  # 生成 1 到 100 的数组
print(test2)
np.savetxt("test2.csv", test2, delimiter=',')

# 假设 y_true 是实际值，y_pred 是预测值
y_true = pd.read_csv('test1.csv').values
print(y_true.shape)
y_pred = pd.read_csv('test2.csv').values
print(y_pred.shape)


# 计算 MAE
mae = np.mean(np.abs(y_true - y_pred))
# 计算 MAPE
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# 计算 RMSE
rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

print(f"MAE: {mae}")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
'''
MAE: 1.0
MAPE: 4.229674260242041
RMSE: 1.0

'''