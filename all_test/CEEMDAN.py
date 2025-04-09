from PyEMD import CEEMDAN
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font", family='Microsoft YaHei')

# 读取已处理的 CSV 文件
df = pd.read_csv('../10 min wind speed data.csv')
# 取风速数据
winddata = df.iloc[:, -1].values

# 可视化
plt.figure(figsize=(15, 5), dpi=100)
plt.grid(True)
plt.plot(winddata, color='green')
plt.show()


# 创建 CEEMDAN 对象
ceemdan = CEEMDAN(trials=100, epsilon=0.005)
# NE=100 epsilon=0.005(信噪比)

# 对信号进行 CEEMDAN分解
IMFs = ceemdan(winddata)

# 可视化
plt.figure(figsize=(20, 15))
plt.subplot(len(IMFs) + 1, 1, 1)
plt.plot(winddata, color='hotpink')
plt.title("原始信号")

for num, imf in enumerate(IMFs):
    plt.subplot(len(IMFs) + 1, 1, num + 2)
    plt.plot(imf, color='c')
    plt.title("IMF " + str(num + 1), fontsize=8)
# 增加第一排图和第二排图之间的垂直间距
plt.subplots_adjust(hspace=0.8, wspace=0.2)
plt.show()

# 分量重构
reconstructed_data = np.sum(IMFs, 0)  # 沿ｙ轴方向求和

plt.figure(figsize=(15, 5))
plt.plot(winddata, linewidth=1, color='hotpink', label='PM2.5')
plt.plot(reconstructed_data, linewidth=1, color='c', label='分解重构结果')
plt.title("CEEMDAN 分解重构结果", fontsize=10, loc='center')
plt.xticks(fontsize=10)  # x 轴刻度字体大小
plt.yticks(fontsize=10)  # y 轴刻度字体大小
plt.legend(loc='upper right')  # 绘制曲线图例，信息来自类型 label
plt.show()

