#!/usr/bin/python3
# -*- coding: utf-8 -*-

'''
@Date    : 2021.11.1
@Author  : Herry
'''
import numpy as np
from vmdpy import VMD
from PyEMD import EMD, EEMD

class Decomposition:
    def __init__(self, data, length):
        self.data = data.reshape(-1)
        self.length = length  # 15

    def SSA(self):
        '''
        奇异谱分析（SSA）方法：
            是一种用于将原始序列分解为可解释子序列，以实现滤波降噪等目的的方法
        基本步骤：
            嵌入：把原始时间序列转化为轨迹矩阵，为后续分析做准备。
            奇异值分解：对轨迹矩阵的协方差矩阵进行分解，得到特征值和特征向量，将轨迹矩阵分解为多个秩为 1 的矩阵之和，每个矩阵包含原始序列不同方面信息。
            分组：将范围分解成无关联的子集，进一步把轨迹矩阵分解为多个子矩阵之和，便于后续分别处理和分析。
            对角平均：把每个子矩阵转换为新序列，最终得到重构序列。实际应用中，选择前m(m<L)个具有较大特征值的分量，能达到滤波降噪的效果。
        Returns
        -------
        '''
        series = self.data
        # step1 嵌入
        windowLen = self.length  # 嵌入窗口长度  15
        seriesLen = len(series)  # 序列长度   2880

        # 构建一个windowLen行、K列的矩阵X，其中K是可生成的子序列数量，矩阵每列是从原序列按窗口滑动截取的子序列
        K = seriesLen - windowLen + 1 # 2866
        X = np.zeros((windowLen, K))  # (15,2866)
        for i in range(K):# 2866    0 - 2865
            X[:, i] = series[i:i + windowLen]
        # step2: svd分解， U和sigma已经按升序排序
        U, sigma, VT = np.linalg.svd(X, full_matrices=False)

        for i in range(VT.shape[0]):
            VT[i, :] *= sigma[i]
        A = VT

        # 重组
        rec = np.zeros((windowLen, seriesLen))  # (15,2880)
        for i in range(windowLen):
            for j in range(windowLen - 1):
                for m in range(j + 1):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= (j + 1)
            for j in range(windowLen - 1, seriesLen - windowLen + 1):
                for m in range(windowLen):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= windowLen
            for j in range(seriesLen - windowLen + 1, seriesLen):
                for m in range(j - seriesLen + windowLen, windowLen):
                    rec[i, j] += A[i, j - m] * U[m, i]
                rec[i, j] /= (seriesLen - j)

        return rec.T

    def EMD(self):
        data = self.data
        decomp = EMD()
        decomp.emd(data)
        imfs, res = decomp.get_imfs_and_residue()
        IMFs = imfs.T
        IMFs = np.insert(IMFs, IMFs.shape[1], values=res, axis = 1)

        return IMFs

    def EEMD(self):
        data = self.data
        decomp = EEMD()
        decomp.eemd(data)
        imfs, res = decomp.get_imfs_and_residue()
        IMFs = imfs.T
        IMFs = np.insert(IMFs, IMFs.shape[1], values=res, axis = 1)

        return IMFs

    def VMD(self):
        data = self.data
        alpha, tau, length, DC, init, tol = 5000, 0, self.length, 0, 1, 1e-8
        u, u_hat, omega = VMD(data, alpha, tau, length, DC, init, tol) # 得到分解模式的集合u、模式的光谱范围u_hat和估计模态的中心频率omega

        return u.T
