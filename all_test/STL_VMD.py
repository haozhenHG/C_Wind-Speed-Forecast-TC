#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2025/3/12 22:12
# @File ：STL_VMD.py
# @Author : 胜天半子
# IDE : PyCharm
# @CSDN : https://blog.csdn.net/HG0724?type=blog
# @GitHub : https://github.com/haozhenHG
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from vmdpy import VMD

def plot_plt(res):
    res.plot()
    plt.tight_layout()
    plt.show()


def stl_vmd_decomposition(data, vmd_k=15):
    """
    STL首次分解 + VMD二次分解
    :param data: 一维时间序列数据
    :param vmd_k: VMD分解的模态数（IMF数量）
    :return: STL分解结果、VMD分解结果
    """
    # 1. STL分解
    stl = STL(data, period=144, robust=True)
    result = stl.fit() # 季节性趋势分解
    trend = result.trend
    seasonal = result.seasonal
    residual = result.resid # 残差是原始时间序列数据减去趋势和季节性成分后剩余的部分，它包含了时间序列中无法由趋势和季节性解释的随机波动或噪声。

    # # 绘图
    # plot_plt(result)
    # plot_plt(result.trend)
    # plot_plt(result.seasonal)
    # plot_plt(result.seasonal)
    # plot_plt(result.resid)


    # 2. VMD分解残差
    alpha = 5000  # 带宽约束参数
    tau = 0.0  # 噪声容忍度（0表示严格保真）
    length = 15
    DC = 0  # 不保留直流分量
    init = 1  # 初始化模式
    tol = 1e-8  # 停止迭代阈值
    u, _, _ = VMD(residual, alpha, tau, vmd_k, DC, init, tol)

    return {
        'stl': {'trend': trend, 'seasonal': seasonal, 'residual': residual},
        'vmd': {'imfs': u}
    }


# 示例运行
if __name__ == '__main__':
    # 1. 读取数据（示例数据为正弦波+噪声）
    # t = np.linspace(0, 100, 1000)
    # data = np.sin(0.1 * np.pi * t) + 0.2 * np.sin(0.5 * np.pi * t) + np.random.normal(0, 0.1, 1000)
    data = pd.read_csv('../10 min wind speed data.csv', header= None) # (2880,3)


    # 2. 执行分解
    decomposition = stl_vmd_decomposition(data.iloc[:,0].values, vmd_k=15)

    # 3. 可视化结果
    plt.figure(figsize=(12, 8))

    # STL分解结果
    plt.subplot(4, 1, 1)
    plt.plot(data, label='Original')
    plt.title('Original Signal')

    plt.subplot(4, 1, 2)
    plt.plot(decomposition['stl']['trend'], label='Trend')
    plt.title('STL Trend')

    plt.subplot(4, 1, 3)
    plt.plot(decomposition['stl']['seasonal'], label='Seasonality')
    plt.title('STL Seasonality')

    plt.subplot(4, 1, 4)
    plt.plot(decomposition['stl']['residual'], label='Residual')
    plt.title('STL Residual')

    plt.tight_layout()
    plt.show()

    # VMD分解结果
    plt.figure(figsize=(18, 12))
    for i, imf in enumerate(decomposition['vmd']['imfs']):
        ax = plt.subplot(decomposition['vmd']['imfs'].shape[0], 1, i + 1)
        line, = ax.plot(imf, label=f'IMF {i + 1}')
        ax.set_title(f'VMD IMF {i + 1}')

        # # 调整图例位置和字体大小
        # ax.legend(handles=[line], loc='center left', bbox_to_anchor=(-0.1, 0.5), fontsize=8)
    plt.tight_layout()
    plt.show()