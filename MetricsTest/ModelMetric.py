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
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# 假设 y_true 是实际值，y_pred 是预测值
y_true = pd.read_csv('../De_data/ssa_denoise_load_720.csv').values
print(y_true.shape)
y_pred = pd.read_csv('../De_data/pre_ssa_tcn_gru.csv').values
print(y_pred.shape)

# 计算 MAE
mae = mean_absolute_error(y_true, y_pred)
mape_ziji = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
mape = mean_absolute_percentage_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print(f"MAE: {mae}")
print(f"MAPE_ziji: {mape_ziji}")
print(f"MAPE: {mape}")
print(f"RMSE: {rmse}")
'''
(719, 1)
(719, 1)
MAE: 1140.793026895511
MAPE: 1.4068390499662535
RMSE: 1455.7335079512443
'''
print('#'*30)
# 归一化数据集
scaler = MinMaxScaler()
data1_normalized = scaler.fit_transform(y_true.reshape(-1, 1)).flatten()
# print(data1_normalized[:10])
'''
[0.69687903 0.55488996 0.73063878 0.92637598 0.53344797 0.53345164
 0.93893919 0.75741552 0.48082726 0.70713074]
'''
data2_normalized = scaler.fit_transform(y_pred.reshape(-1, 1)).flatten()
# print(data2_normalized[:10])

# 计算归一化后的误差指标
mae_after = mean_absolute_error(data1_normalized, data2_normalized)

mask = data1_normalized != 0

mape_ziji_a = np.mean(np.abs((data1_normalized[mask] - data2_normalized[mask]) / data1_normalized[mask])) * 100

# mape_ziji_a = np.mean(np.abs((data1_normalized - data2_normalized) / data1_normalized)) * 100
mape_after = mean_absolute_percentage_error(data1_normalized, data2_normalized)
rmse_after = np.sqrt(mean_squared_error(data1_normalized, data2_normalized))
# 计算 MAE
# mae_after = np.mean(np.abs(y_true - y_pred))
# # 计算 MAPE
# mape_after = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
# # 计算 RMSE
# rmse_after = np.sqrt(np.mean((y_true - y_pred) ** 2))

print(f"归一化后的MAE: {mae_after:.2f}")
print(f"归一化后的mape_ziji_a: {mape_ziji_a:.2f}%")
print(f"归一化后的MAPE: {mape_after:.2f}%")
print(f"归一化后的RMSE: {rmse_after:.2f}")
'''
归一化后的MAE: 0.06
归一化后的MAPE: 0.12%
归一化后的RMSE: 0.07
'''
