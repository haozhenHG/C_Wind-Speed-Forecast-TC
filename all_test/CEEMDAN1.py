import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from PyEMD import CEEMDAN
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def ceemdan(data):
    data = data
    decomp = CEEMDAN()
    imfs = decomp.ceemdan(data)
    print(f"shape of imfs: {imfs.shape}")

    # 可视化
    # plt.figure(figsize=(20, 15))
    # plt.subplot(len(imfs) + 1, 1, 1)
    # plt.plot(data, color='hotpink')
    # plt.title("原始信号")

    # for num, imf in enumerate(imfs):
    #     plt.subplot(len(imfs) + 1, 1, num + 2)
    #     plt.plot(imf, color='c')
    #     plt.title("IMF " + str(num + 1), fontsize=8)
    # # 增加第一排图和第二排图之间的垂直间距
    # plt.subplots_adjust(hspace=0.8, wspace=0.2)
    # plt.show()

    res = data - np.sum(imfs, axis=0)
    IMFs = imfs.T
    print(f"shape of imfs.T: {IMFs.shape}")
    IMFs = np.insert(IMFs, IMFs.shape[1], values=res, axis=1)
    print(f"shape of IMFs: {IMFs.shape}")


    return IMFs


def denoise(data, imfs):
    data = data.reshape(-1) # 将多维数组展平为一维数组
    # print('denoise  ',data.shape)  # denoise  (2880,)
    denoise_data = 0
    for i in range(imfs.shape[1]):  # imfs.shape (2880,15)
        denoise_data += imfs[:,i]
        # print(f"denoise_data shape: {denoise_data.shape}")
        # print(f"data shape: {data.shape}")
        pearson_corr_coef = np.corrcoef(denoise_data, data)
        if pearson_corr_coef[0,1] >=0.995:
            print(i) # 8  前9列数据
            break

    return denoise_data

if __name__ == '__main__':
    # 读取已处理的 CSV 文件
    df = pd.read_csv('../10 min wind speed data.csv')
    # 取风速数据
    winddata = df.iloc[:, -1].values
    print(winddata.shape)
    # ceemdan分解以及可视化
    ceemdan_IMFs = ceemdan(winddata)
    denoise(winddata,ceemdan_IMFs)

'''
每次分解结果不相同  8

'''