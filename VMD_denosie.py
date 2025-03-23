#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/23 19:40
# @File ：VMD_denosie.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
from main import IMF_decomposition, mape
import pandas as pd
import numpy as np


test_number, imfs_number, input_step, pre_step= 720, 6, 24, 1  # 测试一个月的
data = pd.read_csv('jupyter/American_final_to_model.csv', index_col=0)  # (21769, 11)

print(data.shape)
# print(data.iloc[:, -1].values) # 查看负荷列数据
ssa_denoise = IMF_decomposition(data.iloc[:, -1].values, imfs_number)  # 选取负荷列数据  经过降噪之后
# print(ssa_denoise)
# np.savetxt(r'De_data/ssa_denoise_load_720.csv', ssa_denoise[-test_number:], delimiter=',')  # 使用IMFS函数重构成干净数据 ssa_denoise [2880]
# print(mape(data.iloc[:, -1].values, ssa_denoise))  # 0.5965549952390393

# data = pd.read_csv('10 min wind speed data.csv', header= None)
# ssa_denoise = IMF_decomposition(data.iloc[:,2].values, imfs_number)
# print(mape(data.iloc[:,2].values, ssa_denoise))  # 2.1537545055199305


# np.savetxt('ssa_denoise_3.csv', ssa_denoise[-test_number:], delimiter=',')  # 使用IMFS函数重构成干净数据 ssa_denoise [2880]
# pre_emd = single_model(emd_denoise, test_number, 'tcn_gru', input_step, pre_step)
# pre_eemd = single_model(eemd_denoise, test_number, 'tcn_gru', input_step, pre_step)
# pre_vmd = single_model(vmd_denoise, test_number, 'tcn_gru', input_step, pre_step)
